import numpy as np
from tqdm import tqdm


def remove(patch_dict, percent=10):
    return {key:img for (key, img) in tqdm(patch_dict.items()) if enough_color(img, percent)}

def enough_color(img, percent):
    pixel_count = img.shape[0] * img.shape[1]
    black_pixels = np.sum(np.all(img.reshape(-1, 3) == 0, axis=1))
    if 100 * black_pixels / pixel_count >= percent:
        return False
    else:
        return True

