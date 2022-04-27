import os
import shutil
import numpy as np
import torch
import torchio as tio
import h5py
import ffmpeg
import matplotlib.pyplot as plt


def filter_files(key: str, start: str, destination: str, rand_choice: bool = False) -> None:
    """
    whitelist style of filtering, move files with key in filename to dest
    :param key: what you want in a filename
    :param start: starting path of files
    :param destination: where the whitelisted files will go
    :return: None
    """
    count, all_names = 0, []
    print(f'Starting for: {start}')
    for filename in os.listdir(start):
        if key in filename:
            all_names.append(filename)
    print(f'Number of {key} files found: {len(all_names)}')
    if not all_names:
        print(f'No items of {key} were found')
        return None
    elif rand_choice:
        choices = np.random.choice(range(1, len(all_names)), 10, replace=False)
        all_names = set([all_names[i] for i in choices])
    for filename in all_names:
        shutil.move(start + '/' + filename, destination + '/' + filename)
        count += 1
    print(count, all_names)


def random_choice(n, path) -> list:
    with open(path, 'r') as file:
        f = file.read()
    f = f.split('\n')
    nums = np.random.choice(range(1, len(f)), n, replace=False)
    return [f[i] for i in nums]


def clean_text(path: str) -> None:
    with open(path, 'r') as file:
        f = file.read()
    lines = f.split('\n')
    for i in range(len(lines)):
        index = lines[i].find('\t')
        if index == -1:
            continue
        lines[i] = lines[i][:lines[i].find('\t')]
    lines = '\n'.join(lines)
    with open(path, 'w') as file:
        file.write(lines)


def delete_duplicates(path: str) -> None:
    file_names = set()
    count = 0
    for file in os.listdir(path):
        if file in file_names:
            count += 1
            os.remove(path+'/'+file)
        else:
            file_names.add(file)
    print(f'Number of files deleted: {count}')
    return None


def read_h5j(h5j_path):
    channels = list()

    with h5py.File(h5j_path, 'r') as f:
        for c in f['Channels'].keys():
            mp4_path = f'{h5j_path}.{c}.mp4'
            # trick to store the temp file in memory. prob not necessary
            # mp4_path = '/dev/shm/' + mp4_path.split('/')[-1]

            data = f['Channels'][c][:].tofile(mp4_path)
            probe = ffmpeg.probe(mp4_path)
            video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            width = int(video_info['width'])
            height = int(video_info['height'])
            num_frames = int(video_info['nb_frames'])
            pix_fmt = video_info['pix_fmt']
            out, _ = (
                ffmpeg
                    .input(mp4_path)
                    .output('pipe:', format='rawvideo', pix_fmt=pix_fmt)
                    .run(capture_stdout=True, quiet=True)
            )
            os.remove(mp4_path)

            num_bytes = 2
            video = (
                np
                    .frombuffer(out, np.uint8)
                    .reshape([-1, height, width, num_bytes])
            )

            # the 'videos' are stored as 12-bit grayscale apparently
            # here I'm just converting them to float
            assert pix_fmt == 'gray12le', 'unexpected format'
            video = video.astype(np.uint32)
            video = (video[..., 1] << 8) + video[..., 0]
            video = video.astype(np.float32) / (1 << 12)
            # if type(channels) == np.ndarray:
            #             #     channels = channels.tolist()
            #             #     channels.append(video)
            #             # else:
            #             #     channels.append(video)
            channels.append(video)
        # depth, height, width, channels
        channels = np.stack(channels, -1)

    return channels

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

def generate_surrounding_coordinates(coord, is_edge=None):
    coords = np.array(
            [[coord[0] + 1, coord[1] + 1],
            [coord[0] + 1, coord[1]],
            [coord[0], coord[1] + 1],
            [coord[0] - 1, coord[1] - 1],
            [coord[0], coord[1] - 1],
            [coord[0] - 1, coord[1]],
            [coord[0] - 1, coord[1] + 1],
            [coord[0] + 1, coord[1] - 1]])
    if is_edge:
        if is_edge == 'top right':
            coords = np.array([
                    [coord[0] - 1, coord[1] + 1],
                    [coord[0], coord[1] + 1],
                    [coord[0] - 1, coord[1]]])
        elif is_edge == 'top left':
            coords = np.array(
                [[coord[0] + 1, coord[1] + 1],
                [coord[0], coord[1] + 1],
                [coord[0] + 1, coord[1]]])
        elif is_edge == 'bottom left':
            coords = np.array(
                [[coord[0] + 1, coord[1]],
                 [coord[0], coord[1] - 1],
                 [coord[0] + 1, coord[1] - 1]])
        elif is_edge == 'bottom right':
            coords = np.array(
                [[coord[0] - 1, coord[1]],
                    [coord[0], coord[1] - 1],
                    [coord[0] - 1, coord[1] - 1]])
        elif is_edge == 'top':
            coords = np.array(
                [[coord[0] + 1, coord[1]],
                 [coord[0] - 1, coord[1] - 1],
                 [coord[0], coord[1] - 1],
                 [coord[0] - 1, coord[1]],
                 [coord[0] + 1, coord[1] - 1]])
        elif is_edge == 'bottom':
            coords = np.array(
                [[coord[0] + 1, coord[1] + 1],
                 [coord[0] + 1, coord[1]],
                 [coord[0], coord[1] + 1],
                 [coord[0] - 1, coord[1]],
                 [coord[0] - 1, coord[1] + 1]])
        elif is_edge == 'left':
            coords = np.array(
                [[coord[0] + 1, coord[1] + 1],
                 [coord[0] + 1, coord[1]],
                 [coord[0], coord[1] + 1],
                 [coord[0], coord[1] - 1],
                 [coord[0] + 1, coord[1] - 1]])
        elif is_edge == 'right':
            coords = np.array(
                [[coord[0], coord[1] + 1],
                 [coord[0] - 1, coord[1] - 1],
                 [coord[0], coord[1] - 1],
                 [coord[0] - 1, coord[1]],
                 [coord[0] - 1, coord[1] + 1]])
    return coords

if __name__ == '__main__':
    np.random.seed(1)
    rand_choices = random_choice(100, "folder_names.txt")
    [print(i) for i in rand_choices]

