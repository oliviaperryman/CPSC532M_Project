# https://towardsdatascience.com/create-new-animals-using-dcgan-with-pytorch-2ce47810ebd4

import glob
import os
import numpy as np

import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from custom_transforms import (
    normalize,
    random_crop,
    random_jittering_mirroring,
)

"""
Create the pytorch datset for the animal-faces hq and perform average pooling.
"""


class afhqDataset(Dataset):
    def __init__(self, root_dir="../../afhq/train_contours_small"):
        self.root_dir = root_dir
        # originally 64, 64 resizing for saving compute power, changed to 286 for jitter
        self.avgpool = nn.AdaptiveAvgPool2d((286, 286))
        self.all_files = glob.glob(os.path.join(root_dir, "**/*.png"), recursive=True)

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        contour_path = self.all_files[idx]

        # Image
        image_path = contour_path.replace("train_contours_small", "train")
        image_path = image_path.replace("png", "jpg")
        print(image_path)
        image = cv2.imread(image_path) / 255.0
        image = torch.Tensor(image)
        image = image.permute(2, 0, 1)
        image = self.avgpool(image)

        # Contour
        contour = cv2.imread(contour_path) / 255.0
        contour = torch.Tensor(contour)
        contour = contour.permute(2, 0, 1)
        contour = self.avgpool(contour)

        # Label
        label = image_path.split("/")[-2]

        tar = np.array(image).astype(np.float32)
        inp = np.array(contour).astype(np.float32)
        print(inp.shape)
        inp, tar = random_jittering_mirroring(inp, tar)
        inp, tar = normalize(inp, tar)
        # image_a = torch.from_numpy(inp.copy().transpose((2, 0, 1)))
        # image_b = torch.from_numpy(tar.copy().transpose((2, 0, 1)))

        return inp, tar, label

if __name__ == "__main__":
    train_set = afhqDataset()
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)

    batch = next(iter(train_loader))
    print(len(batch))
    print(batch[0].shape)
    print(batch[1].shape)
    print(batch[2])
