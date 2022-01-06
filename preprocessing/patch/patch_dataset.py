import torch
from torch.utils.data import Dataset


class PatchDataset(Dataset):
    def __init__(self, patch_dict):
        """
        Creates a dataset for image patches.
        @param data_folder: path to a folder containing image patches
        """
        self.patch_dict = patch_dict
        self.patch_ids = list(patch_dict.keys())

    def __getitem__(self, idx):
        """
        @returns: (patch_height, patch_width, 3) float array in [0, 1]
        """
        return torch.tensor(self.patch_dict[self.patch_ids[idx]] / 255, dtype=torch.float32)

    def __len__(self):
        return len(self.patch_ids)