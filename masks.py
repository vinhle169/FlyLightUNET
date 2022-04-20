import os
import shutil
import numpy as np
import torch
import torchio as tio
import h5py
import ffmpeg
import skimage.filters as filters
import cv2
import matplotlib.pyplot as plt
from utils import *


def matching_pursuit(image):
    """
    Will a version of matching pursuit to get a mask from an image
    Projects RGB vector on each of the color unit vectors, and then picks the color that maximizes it. Then subtracts
    that color's contribution, and repeat on what's left until no more colors.

    todo: assign each pixel to whatever color its correlated to, do classical matching pursuit, pretend white doesnt happen
    otsu on greyscale and everything below greyscale is backgrnd and then assign colors(matching pursuit on the rest)
    think about identifying boundary pixels, find out strategy for critique
    :return:
    """
    width, height = image.shape[0:2]
    colors = np.array(
        [[1, 1, 1],  # black, grey, white
         [1, 0, 0],  # red
         [1, 0.5, 0.5], # still red
         [0, 1, 0],  # green
         [0.5, 1, 0.5], # still green
         [0, 0, 1],  # blue
         [0.5, 0.5, 1],  # still blue
         [1, 1, 0],  # yellow/orange
         [1, 0, 1],  # purple/pink
         [0, 1, 1]])  # turquoise/cyan
    idx_to_color = {idx: np.array([np.floor(i*255) for i in color], dtype=np.uint8) for idx, color in enumerate(colors)}
    idx_to_color[0] = 0
    colors = colors / np.sqrt((colors ** 2).sum(1, keepdims=True))
    projection = lambda res, col: res @ col.T
    residuals = np.zeros(image.shape)
    num_colors = len(colors)
    deconvolved = np.zeros((width, height, num_colors))
    proj = projection(image, colors)
    # print(proj[15,19,:]) # part of test 2

    best_color = np.argmax(proj, axis=-1)
    for row in range(len(best_color)):
        for col in range(len(best_color[0])):
            residuals[row][col] = idx_to_color[best_color[row][col]]
    residuals = np.array(residuals, dtype=np.uint8)
    return residuals


def greyscale_otsu_threshold(img):
    '''
    :param img: assumes img is a numpy array w/ RGB channels
    :return: img: which has 0 for background and 1 for foreground
    '''
    gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    threshold = filters.threshold_otsu(gray_image)
    mask = gray_image >= threshold
    mask_3d = np.stack((mask, mask, mask), axis=2)
    masked_img = np.where(mask_3d == 1, img, mask_3d)
    return masked_img


def determine_boundary(img):
    '''
    :param img:
    :return: 2d array with 0 for background 1 for foreground 2 for boundary
    '''
    def edge_checker(img, coordinate):
        '''
        :param img:
        :param coordinate: tuple(x,y)
        :return: type of edge
        '''
        max_x, max_y = img.shape[0:2]
        if coordinate[0] == max_x and coordinate[1] == max_y:
            return 'bottom right'
        elif coordinate[0] == 0 and coordinate[1] == 0:
            return 'top left'
        elif coordinate[0] == max_x and coordinate[1] == 0:
            return 'top right'
        elif coordinate[0] == 0 and coordinate[1] == max_y:
            return 'bottom left'
        elif coordinate[0] == max_x:
            return 'right'
        elif coordinate[0] == max_y:
            return 'bottom'
        elif coordinate[0] == 0:
            return 'left'
        elif coordinate[0] == 0:
            return 'top'
        return None

    def boundary_checker(img, coordinate):
        '''
        :param img:
        :param coordinate: tuple (x,y)
        rules for boundary (to be edited, since coloring scheme in matching pursuit needs work to be done):
        if fully surrounded by background
        if on the left there's background and on the right there's foreground
        if on the right ^^
        if on the top ^^
        if on the bottom ^^
        if 3/4th surrounded by background
        :return: true if boundary else false
        '''
        is_edge = edge_checker(img, coordinate)
        if is_edge:
            pass
        else:
            surrounding = generate_surrounding_coordinates(coordinate)
            colors = np.array([img[i[0]][i[1]] for i in surrounding])
            return not np.all(colors)
    background = []
    boundary = []
    foreground = []
    blank = np.zeros(img.shape)
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            if np.any(img[row][col]):
                if boundary_checker(img, (row, col)):
                    boundary.append((row,col))
                else:
                    foreground.append((row,col))
            else:
                background.append((row,col))
    for i in background:
        blank[i[0]][i[1]] = [0,0,0]
    for i in foreground:
        blank[i[0]][i[1]] = [0,1,0]
    for i in boundary:
        blank[i[0]][i[1]] = [2,0,0]
    return blank


if __name__ == "__main__":
    # img = cv2.imread('flylight_logo.png')
    desired_width = 320
    np.set_printoptions(linewidth=desired_width)

    img = cv2.imread('test_image.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = img[200:220, 115:135, :] # test_1
    # img = img[140:160, 100:120, :] # test_2
    # print(img[15,19,:]) # part of test 2
    masked_image = greyscale_otsu_threshold(img)
    # convert foreground to hsv and run edge detector on hue channel
    # add boundary with colors
    # otsu threshold -> foreground -> convert to hsv -> run edge detector on hue channel -> otsu output
    # try on actual images
    # try various scikit image edge detectors
    image_match = matching_pursuit(masked_image)
    boundaries = determine_boundary(image_match)
    boundaries = np.array(boundaries) * 100

    plt.imshow(boundaries)
    plt.show()
    # fig, axs = plt.subplots(1,2)
    # axs[0].imshow(masked_image)
    # axs[0].title.set_text('Post Otsu Image')
    # axs[1].imshow(image_match)
    # axs[1].title.set_text('Post Matching Pursuit')
    # plt.savefig('results/test_whole.png')
    # plt.show()
