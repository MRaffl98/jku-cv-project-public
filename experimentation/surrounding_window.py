import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from preprocessing.patch.patch_generator import clear, create
from preprocessing.patch.black_patch_removal import remove


"""
The idea here is to subtract for a given patch the average R, G, and B values from a surrounding window.
Currently, for a given patch P the average is calculated based on a surrounding window not directly adjacent to
patch P, i.e. there is always a patch in between, which is sometimes called guard window.
In the following illustration, the average subtracted from patch P (e.g. 16x16) is computed based on all Y patches.

Y Y Y Y Y
Y N N N Y
Y N P N Y
Y N N N Y
Y Y Y Y Y

The goal is to increase small anomalies (such as humans) where the surrounding window is rather different from 
the patch containing the human. On the other hand large anomalies (such as beacons) should be removed since
ideally the surrounding window still contains a part of it and then subtracts that. 
"""


width = 16
k = 1024 / 16


if __name__ == '__main__':

    #### 1. CREATE PATCHES OF 1 VALIDATION SAMPLE (make sure the folder 'image_patches/validation' exists)
    clear('validation') # delete current patches
    create('validation', n_samples=1, subimage_size=width, integrate=True, mask_timestep=True) # create new patches
    remove('validation', percent=1) # remove mostly black patches


    #### 2. FOR EACH PATCH SUBTRACT THE AVERAGE R,G,B VALUES OF ITS SURROUNDING PATCHES
    # (this is particularly annoying for border patches)
    filenames = os.listdir(os.path.join(os.getenv("DATA_PATH"), "../image_patches", "validation"))

    for file in tqdm(filenames):
        number = int(file.split('_')[-1][:-4])
        surrounding_patch_numbers = []

        # patches two rows above and below in same column
        surrounding_patch_numbers.append(number - 2*k)
        surrounding_patch_numbers.append(number + 2*k)

        # patches two columns to the right
        if (number+1) % k <= (k-2):
            surrounding_patch_numbers.append(number + 2)
            surrounding_patch_numbers.append(number + 2 + k)
            surrounding_patch_numbers.append(number + 2 - k)
            surrounding_patch_numbers.append(number + 2 + 2*k)
            surrounding_patch_numbers.append(number + 2 - 2*k)

        # patches one column to the right
        if (number+1) % k <= (k-1):
            surrounding_patch_numbers.append(number + 1 + 2*k)
            surrounding_patch_numbers.append(number + 1 - 2*k)

        # patches two columns to the left
        if (number+1) % k > 2:
            surrounding_patch_numbers.append(number - 2)
            surrounding_patch_numbers.append(number - 2 + k)
            surrounding_patch_numbers.append(number - 2 - k)
            surrounding_patch_numbers.append(number - 2 + 2*k)
            surrounding_patch_numbers.append(number - 2 - 2*k)

        # patches one column to the left
        if (number+1) % k > 1:
            surrounding_patch_numbers.append(number - 1 + 2*k)
            surrounding_patch_numbers.append(number - 1 - 2*k)

        # get filenames of actually existing surrounding patches
        filename_start = ''.join(file.split('_')[:-1])
        surrounding_filenames = [filename_start + '_' + str(int(x)) + '.png' for x in surrounding_patch_numbers]
        surrounding_filenames = list(set(filenames).intersection(surrounding_filenames))

        # average over the surrounding patches to end up with a mean R, G and B value
        surrounding_patches = [cv.imread(os.path.join(os.getenv("DATA_PATH"), "../image_patches", "validation", f)) for f in surrounding_filenames]
        mean_surrounding = np.mean((surrounding_patches), axis=(0, 1, 2))

        # subtract average R,G,B values computed based on the surrounding filenames
        # from the current patch and write the page to a file again: 'new_' + old_filename
        patch = cv.imread(os.path.join(os.getenv("DATA_PATH"), "../image_patches", "validation", file))
        new_patch = np.abs(patch - mean_surrounding).astype(np.int8)
        cv.imwrite(os.path.join(os.getenv("DATA_PATH"), "../image_patches", "validation", 'new_' + file), new_patch)


    #### 3. RECONSTRUCT THE FULL IMAGE BASED ON THE MODIFIED SAMPLES AGAIN AND MANUALLY EXPLORE THE RESULT
    full_images = {str(t):np.zeros((1024, 1024, 3), dtype=np.int8) for t in range(7)}

    for file in tqdm(['new_' + f for f in filenames]):
        t = file.split('_')[-2][-1]
        number = int(file.split('_')[-1][:-4])
        row = number // k
        col = number % k
        patch = cv.imread(os.path.join(os.getenv("DATA_PATH"), "../image_patches", "validation", file))
        full_images[t][int(width*row):int(width*row+width), int(width*col):int(width*col+width), :] += patch

    plt.imshow(full_images['3'])
    plt.show()