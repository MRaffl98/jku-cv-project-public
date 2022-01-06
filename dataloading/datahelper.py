import os
import json
import cv2 as cv


def get_sample_paths(path, subset='train'):
    """
    Returns a list of paths.
    @param path: path to the data folder
    @param subset: one of ('train', 'validation', 'test')
    @return: list containing paths to all samples in the subset
    """
    path = os.path.join(path, subset)
    paths = [os.path.join(path, sample) for sample in os.listdir(path) if sample!='labels.json']
    return paths


def load_sample(path):
    """
    Load a sample in specified path.
    @param path: path to a specific sample
    @return: (image_dict, homography_dict) tuple, where the dict keys are '0-B01', ..., '6-G05'
    """
    files = [f for f in os.listdir(path)]
    img_files = [f for f in files if f.endswith('png')]
    images = {f[:5]:cv.imread(os.path.join(path, f)) for f in img_files}
    homographies = json.load(open(os.path.join(path, 'homographies.json')))
    return images, homographies


def get_images_at_timestep(images, homographies, timestep):
    """
    Returns all images of a timestep for some sample.
    @param images: dict with images from a sample
    @param homographies: dict with homographies from the same sample
    @param timestep: int in (0, ..., 6)
    @returns: (image_dict, homography_dict) tuple, where keys are '{timestep}-B01', ..., '{timestep}-G05'
    """
    assert 0 <= timestep <= 6
    filename_stubs = ["-B01", "-B02", "-B03", "-B04", "-B05", "-G01", "-G02", "-G03", "-G04", "-G05"]
    images_ = {}
    homographies_ = {}
    for filename_stub in filename_stubs:
        file_ident = str(timestep) + filename_stub
        images_[file_ident] = images[file_ident]
        homographies_[file_ident] = homographies[file_ident]
    assert len(images_) == 10
    assert len(homographies_) == 10
    return images_, homographies_


def load_timestemp_mask():
    """
    @returns: mask.png image
    """
    return cv.imread(os.path.join(os.getenv("DATA_PATH"), "mask.png"))


def get_center_image(path):
    """
    @param path: path to specific sample
    @returns: image 3-B01 for the given sample
    """
    return cv.imread(os.path.join(path, "3-B01.png"))


def load_labels(path):
    """
    @path: path to some folder containing a labels.json file
    @returns: labels.json
    """
    return json.load(open(os.path.join(path, "labels.json")))


if __name__=='__main__':
    validation_paths = get_sample_paths(subset='validation')
    images, homographies = load_sample(validation_paths[0])
    validation_set = [load_sample(path) for path in validation_paths]