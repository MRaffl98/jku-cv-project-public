import numpy as np
from tqdm import tqdm


def remove_light(patch_dict):
    return {key:img for (key, img) in tqdm(patch_dict.items()) if not too_much_color(img)}

def too_much_color(img):
    light_pixels = np.sum(np.any(img.reshape(-1, 3) > 250, axis=1))
    return light_pixels > 0

