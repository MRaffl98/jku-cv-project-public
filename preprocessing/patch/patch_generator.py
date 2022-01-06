
def split_image(image, patch_size):
    """
    Converts an image into multiple patches.
    @param image: an image to be divided into patches
    @param subimage_size: width and height of each patch
    @returns: a list of image patches
    """
    subimages = []
    w, h = patch_size, patch_size
    if 1024 % w != 0 or 1024 % h != 0:
        raise ValueError("Subimage size is not valid.")

    for x in range(int(1024 / w)):
        for y in range(int(1024 / h)):
            subimages.append(image[x * w:x * w + w, y * h:y * h + h])
    return subimages


def fill_patch_dict(patch_dict, img, sample_id, camera, patch_size):
    subimages = split_image(img, patch_size)
    for (subimage_counter, subimg) in enumerate(subimages):
        patch_dict[sample_id + '-' + camera + '-' + str(subimage_counter)] = subimg
    return patch_dict