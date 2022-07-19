import os

import cv2
import numpy as np
from imageio import imread
from torch.utils.data import Dataset
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Resize([1024, 256]),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


def sample_within_bounds(signal, x, y, bounds):
    xmin, xmax, ymin, ymax = bounds

    idxs = (xmin <= x) & (x < xmax) & (ymin <= y) & (y < ymax)

    sample = np.zeros((x.shape[0], x.shape[1], signal.shape[-1]))
    sample[idxs, :] = signal[x[idxs], y[idxs], :]

    return sample


def sample_bilinear(signal, rx, ry):
    signal_dim_x = signal.shape[0]
    signal_dim_y = signal.shape[1]

    # obtain four sample coordinates
    ix0 = rx.astype(int)
    iy0 = ry.astype(int)
    ix1 = ix0 + 1
    iy1 = iy0 + 1

    bounds = (0, signal_dim_x, 0, signal_dim_y)

    # sample signal at each four positions
    signal_00 = sample_within_bounds(signal, ix0, iy0, bounds)
    signal_10 = sample_within_bounds(signal, ix1, iy0, bounds)
    signal_01 = sample_within_bounds(signal, ix0, iy1, bounds)
    signal_11 = sample_within_bounds(signal, ix1, iy1, bounds)

    na = np.newaxis
    # linear interpolation in x-direction
    fx1 = (ix1 - rx)[..., na] * signal_00 + (rx - ix0)[..., na] * signal_10
    fx2 = (ix1 - rx)[..., na] * signal_01 + (rx - ix0)[..., na] * signal_11

    # linear interpolation in y-direction
    return (iy1 - ry)[..., na] * fx1 + (ry - iy0)[..., na] * fx2


def polar_transform(idx):
    idx = 'air{}.jpg'.format(str(int(idx)))
    S = 1200  # Original size of the aerial image
    height = 128  # Height of polar transformed aerial image
    width = 512  # Width of polar transformed aerial image

    i = np.arange(0, height)
    j = np.arange(0, width)
    jj, ii = np.meshgrid(j, i)

    y = S / 2. - S / 2. / height * (height - 1 - ii) * np.sin(2 * np.pi * jj / width)
    x = S / 2. + S / 2. / height * (height - 1 - ii) * np.cos(2 * np.pi * jj / width)

    input_dir = 'dataset/dataset_XJTU/air'
    # output_dir = 'dataset_XJTU/polarmap/'
    signal = imread(os.path.join(input_dir, idx))
    signal = cv2.resize(signal, (S, S), interpolation=cv2.INTER_AREA)
    image = sample_bilinear(signal, x, y)
    image = np.array(image, dtype=np.uint8)
    return transform(image).unsqueeze(0).double()


def ground_transform(idx):
    input_dir = 'dataset/dataset_XJTU/ground'
    idx = 'ground{}.jpg'.format(str(int(idx)))
    # output_dir = 'dataset_XJTU/streetview/'
    signal = imread(os.path.join(input_dir, idx))
    start = int(signal.shape[1] / 4)
    image = signal[:, start: start + int(start / 2), :]
    image = cv2.resize(image, (1024, 256), interpolation=cv2.INTER_AREA)
    image = np.array(image, dtype=np.uint8)
    return transform(image).unsqueeze(0).double()


class innoDataset(Dataset):
    def __init__(self, ground_transform=ground_transform, air_transform=polar_transform):
        self.ground_img = r'dataset/dataset_XJTU/ground'
        self.air_img = r'dataset/dataset_XJTU/air'
        self.ground_transform = ground_transform
        self.air_transform = air_transform
        self.img_labels = range(len(os.listdir(self.ground_img)))
        # self.is_ground = is_ground

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        ground_image = None
        air_image = None
        air_image_other = None
        ground_image = self.ground_transform(idx)
        air_image = self.air_transform(idx)
        # if self.ground_transform and self.is_ground:
        #     ground_image = self.ground_transform(idx)
        #     return ground_image
        # if self.air_transform and not self.is_ground:
        #     air_image = self.air_transform(idx)
        #     return air_image
        while 1:
            rand_idx = int(np.random.randint(low=0, high=self.__len__()))
            if rand_idx != idx:
                air_image_other = self.air_transform(rand_idx)
                break
            else:
                continue
        return (ground_image, air_image, air_image_other, idx)
