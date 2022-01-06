import os
import cv2 as cv
import matplotlib.pyplot as plt

from dataloading.datahelper import get_sample_paths, load_sample, load_labels
from preprocessing.image.integrate_frames import warp_timestep, integrate_images


if __name__ == '__main__':
    # handle path
    validation_paths = get_sample_paths(os.getenv('DATA_PATH'), subset='validation')
    # load and integrate images of central timestep
    images, homographies = load_sample(validation_paths[-1])
    warped_images = warp_timestep(images, homographies, timestep=3)
    integrated_image = integrate_images(warped_images)
    # load and draw bounding boxes
    labels = load_labels(validation_paths[-1])
    for bb in labels:
        x, y, w, h = bb
        cv.rectangle(integrated_image, (x, y), (x + w, y + h), (0, 0, 255), 5)
    # plot warped image
    plt.imshow(cv.cvtColor(integrated_image, cv.COLOR_BGR2RGB))
    plt.show()


