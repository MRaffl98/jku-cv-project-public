import cv2 as cv
import numpy as np


def patch2image(patch_numbers, values, patch_size):
    imagesize = patch_size**2
    positions = patch_numbers
    indices = np.arange((1024*1024) / imagesize)
    losses = np.zeros_like(indices)
    losses[positions] = values
    # reshape vector to image form
    new_image = losses.reshape(int(1024/patch_size), int(1024/patch_size))
    new_image = cv.resize(new_image, (0,0), fx=patch_size, fy=patch_size)
    return new_image