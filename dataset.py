import os
import h5py
import ffmpeg
import torch
import numpy as np
import torchvision
from utils import *
import numpy.random as npr
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from imgaug import augmenters as iaa
from sklearn.model_selection import train_test_split


class FlyLight(torch.utils.data.Dataset):

    def __init__(self, paths, transforms: iaa.Sequential = None) -> None:
        """
        :param path: path to the folder containing all the h5j files
        :param transforms: optional transforms from the imgaug python package
        :return: None
        """
        print("Initializing data conversion and storage...")
        assert paths is not None, \
            "Path to data folder is required"
        super(FlyLight).__init__()
        self.paths = paths
        self.transforms = transforms
        print("Done")

    def __getitem__(self, idx):
        """
        :param train_set: boolean where True means use the train set, False means test set, and None means entire set
        :param idx: index to index into set of data
        :param transform: boolean to decide to get transformed data or not
        :return: h5j image stack now as numpy array
        """
        print("Grabbing h5j file and converting to numpy array...")
        idx_path = self.paths[idx]
        stack = read_h5j(idx_path)
        images = [image[..., :-1] for image in stack]
        print("Performing image augmentation...")
        images = self.transforms(images=images)
        print("Done")
        return torch.tensor(np.array(images))

    def __len__(self):
        return len(self.paths)


class FlyLightDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size: int = 1, transforms: iaa.Sequential = None, test_size: float = .3):
        super().__init__()
        self.transforms = transforms
        self.test_size = test_size
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.items = [self.data_dir + filename for filename in os.listdir(self.data_dir)]
        assert data_dir is not None, \
            "Path to data folder is required"
        self.setup('fit')

    def setup(self, stage):
        self.train, self.test = train_test_split(self.items, test_size=self.test_size)
        if stage in (None, "test"):
            self.fly_test = FlyLight(self.test)
        if stage in (None, "fit"):
            self.fly_train = FlyLight(self.train, transforms=self.transforms)


    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.fly_train, batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.fly_test, batch_size=self.batch_size)

if __name__ == '__main__':
    torch.cuda.empty_cache()
    r = npr.RandomState(1234)
    seq = iaa.Sequential([iaa.Crop(px=(0, 16)), iaa.Fliplr(0.5), iaa.GaussianBlur(sigma=(0, 3.0))])

    fldm = FlyLightDataModule(data_dir='rand_files/', batch_size=1, transforms=seq, test_size=0.3)
    x = fldm.train_dataloader()
    count = 0
    for batch in x:
        print(batch[0].shape)
        count += 1
        if count >= 2:
            break


