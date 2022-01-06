import numpy as np
import cv2 as cv
from functools import reduce


def warp_timestep(images, homographies, timestep):
    keys = [key for key in images.keys() if key[0] == str(timestep)]
    warped_images = []
    for key in keys:
        image = images[key]
        homography = np.array(homographies[key])
        warped_images.append(cv.warpPerspective(image, homography, image.shape[:2]))
    return warped_images


def integrate_images(images):
    # compute image sum
    images = [img.astype(np.uint16) for img in images]
    image_sum = reduce(np.add, images)
    # compute image counter
    masks = [(img > 0).astype(np.uint8) for img in images]
    counter = reduce(np.add, masks)
    counter[counter==0] = 1
    # zero out all regions where less than 10 images contribute
    image_sum = np.where(counter==10, image_sum, 0)
    # compute mean
    average_image = image_sum / counter
    return average_image.astype(np.uint8)