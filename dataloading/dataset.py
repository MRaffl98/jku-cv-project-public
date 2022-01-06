import cv2
import numpy
import random

from dataloading.datahelper import get_sample_paths, load_sample, load_timestemp_mask
from preprocessing.image.integrate_frames import warp_timestep, integrate_images
from torch.utils.data import Dataset


class WisarDataset(Dataset):
    def __init__(self, base_dir, subset="train",
                 integrate=False, mask_timestemp=False, random_rotation=False):
        """
        Creates a dataset out of an image folder.
        @param base_dir: Path to data directory
        @param integrate: Whether or not images should be integrated. Note that the homographies are not returned in this case.
        @param normalize: Whether or not image channels should be normalized
        @param mask_timestamp: Whether or not the timestamps should be masked
        @param random_rotation: Whether or not a random rotation should be applied to all images in a sample
        @param subset: One of ('train', 'validation', 'test')
        """
        self.base_dir = base_dir
        self.samples = get_sample_paths(base_dir, subset)
        self.integrate = integrate
        self.mask_timestemp = mask_timestemp
        self.random_rotation = random_rotation

    def __getitem__(self, idx):
        """
        @returns: (image_dict, homography_dict) tuple, if self.integrate is True, homography_dict will be None
        """
        sample = load_sample(self.samples[idx])

        if self.random_rotation:
            rotation = random.choice([cv2.ROTATE_180, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE])
            for k, v in sample[0].items():
                sample[0][k] = cv2.rotate(sample[0][k], rotation)

        if self.mask_timestemp:
            mask = load_timestemp_mask()
            if self.random_rotation:
                mask = cv2.rotate(mask, rotation)
            for k, v in sample[0].items():
                sample[0][k] = numpy.bitwise_and(v, mask)

        if self.integrate:
            warped_images = []
            for t in range(7):
                warped_images.append(integrate_images(warp_timestep(sample[0], sample[1], timestep=t)))
            sample = {(str(t) + '-integrated'):img for t, img in enumerate(warped_images)}

        return (sample, None) if self.integrate else sample

    def __len__(self):
        return len(self.samples)


def identity_collator(batch):
    """
    Function to collate samples of a batch.
    """
    return batch