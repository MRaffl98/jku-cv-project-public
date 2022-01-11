import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from functools import reduce

from dataloading.datahelper import get_sample_paths, load_sample
from preprocessing.image.integrate_frames import warp_timestep

#### LOADING
path = os.getenv('DATA_PATH')
sample_paths = get_sample_paths(path, subset='validation')
images, homographies = load_sample(sample_paths[0])
print(sample_paths[0])

#### WARPING
timestep = 3
identifiers = [key for key in images.keys() if key[0] == str(timestep)]
warped_images = warp_timestep(images, homographies, timestep)

for i in range(len(warped_images)):
    print(f'image {i}')
    for c in range(3):
        print(f'channel {c}')
        with np.printoptions(threshold=np.inf):
            print(warped_images[i][515:555, 534:578, c])

with np.printoptions(threshold=np.inf):
    print(warped_images[3][515:555, 534:578, 1])

plt.imshow(cv.cvtColor(warped_images[3][515:555, 534:578, :], cv.COLOR_BGR2RGB))

#### OLD INTEGRATION (MEAN)
# compute image sum
warped_images_16 = [img.astype(np.uint16) for img in warped_images]
image_sum = reduce(np.add, warped_images_16)
# compute image counter
masks = [(img > 0).astype(np.uint8) for img in warped_images_16]
counter = reduce(np.add, masks)
counter[counter == 0] = 1
# zero out all regions where less than 10 images contribute
image_sum = np.where(counter == 10, image_sum, 0)
# compute mean
average_image = image_sum / counter
average_image = average_image.astype(np.uint8)

plt.imshow(cv.cvtColor(average_image, cv.COLOR_BGR2RGB))

#### MAXIMUM INTEGRATION
# compute image sum
image_max = reduce(np.maximum, warped_images)
# compute image counter
masks = [(img > 0).astype(np.uint8) for img in warped_images]
counter = reduce(np.add, masks)
counter[counter == 0] = 1
# zero out all regions where less than 10 images contribute
image_max = np.where(counter == 10, image_max, 0)

plt.imshow(cv.cvtColor(image_max, cv.COLOR_BGR2RGB))