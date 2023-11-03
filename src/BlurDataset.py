import os
import torch
import pandas as pd
#from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import h5py

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

class BlurDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, h5_file, root_dir, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(h5_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks], dtype=float).reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample

def read_h5(filename):

    with h5py.File(filename, "r") as f:
        # Print all root level object names (aka keys) 
        # these can be group or dataset names 
        print("Keys: %s" % f.keys())
        key_list = list(f.keys())
        for i, a_group_key in enumerate(key_list):
            if a_group_key=='coordinate':
                coords = list(f[a_group_key])
            elif a_group_key=='info':
                info = list(f[a_group_key])
            elif a_group_key=='groundtruth':
                gt = to_torch
            else:
                mono = to_torch
            break
        ds_arr = f[a_group_key][()]  # returns as a numpy array
        
    return gt, mono, coords, info