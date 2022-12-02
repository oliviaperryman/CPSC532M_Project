import cv2
import numpy as np
import torch

"""Data Preprocessing"""


def random_crop(image, dim):
    height, width, _ = dim
    x, y = np.random.uniform(low=0, high=int(height - 256)), np.random.uniform(
        low=0, high=int(width - 256)
    )
    return image[:, int(x) : int(x) + 256, int(y) : int(y) + 256]


def random_jittering_mirroring(input_image, target_image, height=286, width=286):

    # remove resizing to 286x286 if input and target is given as 286
    # input_image = cv2.resize(
    #     input_image, (height, width), interpolation=cv2.INTER_NEAREST
    # )
    # target_image = cv2.resize(
    #     target_image, (height, width), interpolation=cv2.INTER_NEAREST
    # )

    # cropping (random jittering) to 256x256
    print(input_image.shape)
    print(target_image.shape)
    print("_______")
    stacked_image = np.stack([input_image, target_image], axis=0)
    cropped_image = random_crop(stacked_image, dim=[256, 256, 3])

    input_image, target_image = cropped_image[0], cropped_image[1]
    print(input_image.shape)
    print(target_image.shape)
    print("_______")
    # print(input_image.shape)
    if torch.rand(()) > 0.5:
        # random mirroring
        input_image = np.fliplr(input_image)
        target_image = np.fliplr(target_image)
    return input_image, target_image


def normalize(inp, tar):
    input_image = (inp / 127.5) - 1
    target_image = (tar / 127.5) - 1
    return input_image, target_image
