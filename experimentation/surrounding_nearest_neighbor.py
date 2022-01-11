import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from tqdm import trange
from functools import reduce

from dataloading.datahelper import get_sample_paths, load_sample
from preprocessing.image.integrate_frames import warp_timestep

#### LOADING
path = os.getenv('DATA_PATH')
sample_no = 3
sample_paths = get_sample_paths(path, subset='validation')
images, homographies = load_sample(sample_paths[sample_no])
print(sample_paths[sample_no])

#### WARPING
timestep = 3
identifiers = [key for key in images.keys() if key[0] == str(timestep)]
warped_images = warp_timestep(images, homographies, timestep)

#### INTEGRATION
def maximum_integration(warped_images):
    # compute image sum
    image_max = reduce(np.maximum, warped_images)
    # compute image counter
    masks = [(img > 0).astype(np.uint8) for img in warped_images]
    counter = reduce(np.add, masks)
    counter[counter == 0] = 1
    # zero out all regions where less than 10 images contribute
    image_max = np.where(counter == 10, image_max, 0)
    return image_max

integrated_image = maximum_integration(warped_images)
plt.imshow(integrated_image)

i = 500
j = 400

guard = 100
window = 1
total = guard + window

padded_image = np.pad(integrated_image, [(total, total), (total, total), (0, 0)]).astype(np.int16)
new_image = np.zeros((1024, 1024, 3), dtype=np.int16)

for i in trange(total+400, total+700): # trange(total, total+1024)
    for j in range(total+400, total+700): # range(total, total+1024)
        # target pixel
        pixel = padded_image[i, j, :].reshape(-1, 3)
        # surrounding window with guard window in between
        left = padded_image[i-total:i+total, j-total:j-guard, :].reshape(-1, 3)
        right = padded_image[i-total:i+total, j+guard:j+total, :].reshape(-1, 3)
        top = padded_image[i-total:i-guard, j-guard:j+guard, :].reshape(-1, 3)
        bottom = padded_image[i+guard:i+total, j-guard:j+guard, :].reshape(-1, 3)
        neighborhood = np.vstack((left, right, top, bottom))
        # nearest neighbor search
        kdtree = KDTree(neighborhood)
        d, idx = kdtree.query(pixel, k=100)
        # resulting pixel difference
        new_image[i-total, j-total, :] = pixel - np.mean(neighborhood[idx.ravel()], axis=0)


