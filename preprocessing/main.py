import os
import cv2 as cv
import matplotlib.pyplot as plt

from dataloading.datahelper import get_sample_paths, load_sample
from preprocessing.image.integrate_frames import warp_timestep, integrate_images


if __name__ == '__main__':
    #### 0. Load example data first
    validation_paths = get_sample_paths(os.getenv('DATA_PATH'), subset='validation')
    images, homographies = load_sample(validation_paths[0])
    print(validation_paths[0])
    plt.imshow(cv.cvtColor(images['0-B04'], cv.COLOR_BGR2RGB))
    plt.show()

    #### 1. Example how to warp a timestep
    warped_images = warp_timestep(images, homographies, timestep=3)
    plt.imshow(cv.cvtColor(warped_images[0], cv.COLOR_BGR2RGB))
    plt.show()

    #### 2. Example how to integrate warped images of a timestep
    integrated_image = integrate_images(warped_images)
    plt.imshow(cv.cvtColor(integrated_image, cv.COLOR_BGR2RGB))
    plt.show()