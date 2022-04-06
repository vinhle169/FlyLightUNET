import os
import shutil
import numpy as np
import torch
import torchio as tio
import h5py
import ffmpeg
import cv2
import matplotlib.pyplot as plt


def create_masks(image, color_order = [0, 1, 2, 3, 4, 5, 6]):
    """
    Will a version of matching pursuit to get a mask from an image
    Projects RGB vector on each of the color unit vectors, and then picks the color that maximizes it. Then subtracts
    that color's contribution, and repeat on what's left until no more colors.
    :return:
    """
    width, height = image.shape[0:2]
    colors = np.array(
        [[1, 1, 1],  # black, grey, white
         [1, 0, 0],  # red
         [0, 1, 0],  # green
         [0, 0, 1],  # blue
         [1, 1, 0],  # yellow/orange
         [1, 0, 1],  # purple/pink
         [0, 1, 1]])  # turquoise/cyan
    # unit_colors = colors / np.linalg.norm(colors)
    colors = colors / np.sqrt((colors ** 2).sum(1, keepdims=True))
    print(colors)
    projection = lambda res, col: res @ col.T
    num_colors = colors.shape[0]
    residuals = image.copy()
    deconvolved = np.zeros((width, height, num_colors))
    for color in color_order:
        proj = projection(residuals, colors)
        this_color_best = np.argmax(proj, axis=-1) == color
        deconvolved[..., color] = proj[..., color] * this_color_best
        residuals = (residuals - deconvolved[...,[color]] @ colors[[color]])
        residuals = residuals.clip(0, np.inf)
    error = ((img - deconvolved @ colors)**2).mean()
    return error, deconvolved, colors

if __name__ == "__main__":
    img = cv2.imread('flylight_logo.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img[150:170, 200:220, :]
    img = np.concatenate((img, np.zeros((20,20,3), dtype=np.uint8)), axis=0)
    plt.imshow(img)
    plt.show()
    error, img2, colors = create_masks(img)
    # plt.imshow(img2 @ colors)
    # plt.show()
    fig, axs = plt.subplots(2, 3, figsize=(21, 14))
    print(img2.shape)
    axs = axs.ravel()
    for k in range(7):
        axs[k].imshow(img2[..., [k]] * colors[k])  # multiplying just to make it visible
    plt.show()

