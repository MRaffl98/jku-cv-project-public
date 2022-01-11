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


def reconstruct(patch_numbers, patches, patch_size):
    new_image = np.zeros((1024, 1024, 3))
    w, h = patch_size, patch_size
    if 1024 % w != 0 or 1024 % h != 0:
        raise ValueError("Subimage size is not valid.")
    for x in range(int(1024 / w)):
        for y in range(int(1024 / h)):
            patch_idx = x * (1024 / w) + y
            if patch_idx in patch_numbers:
                patch = patches[patch_numbers == patch_idx].reshape(w, h, 3)
                new_image[x * w:x * w + w, y * h:y * h + h] += patch
    return new_image.astype('float')