import numpy as np
import scipy
import h5py

import os
from os import listdir
from os.path import splitext, isfile, join
import torch
import numpy as np
from PIL import Image as Image
from DataAug import PairCompose, PairRandomCrop, PairRandomHorizontalFilp, PairToTensor
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader

from matplotlib import pyplot as plt
import seaborn as sn
from tqdm import tqdm

import logging
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from pathlib import Path
from utils.GetLowestGPU import GetLowestGPU

device = torch.device(GetLowestGPU(pick_from=[0,1,2,3]))
def to_torch(x):
    return torch.from_numpy(x).float().to(device)
def to_numpy(x):
    return x.detach().cpu().numpy()

def load_h5(filename="data/Dataset.h5"):
    h5 = h5py.File(filename,'r')

    GT_data = h5['groundtruth']
    mono_data = h5['mono']
    coord_data = h5['coordinate']
    info_data = h5['info']

def h5_dataloader(path, batch_size=64, num_workers=0, use_transform=True):
    image_dir = os.path.join(path, 'Dataset.h5')

    transform = None
    if use_transform:
        transform = PairCompose(
            [
                PairRandomCrop(256),
                PairRandomHorizontalFilp(),
                PairToTensor()
            ]
        )
    dataloader = DataLoader(
        DeblurDataset(image_dir, transform=transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader

def train_dataloader(path, batch_size=64, num_workers=0, use_transform=True):
    image_dir = os.path.join(path, 'train')

    transform = None
    if use_transform:
        transform = PairCompose(
            [
                PairRandomCrop(256),
                PairRandomHorizontalFilp(),
                PairToTensor()
            ]
        )
    dataloader = DataLoader(
        DeblurDataset(image_dir, transform=transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader

class DeblurDataset(Dataset):
    def __init__(self, h5_filename='Dataset', root_dir='data', transform=None, is_test=False):
        self.image_dir = root_dir
        self.filepath = os.listdir(os.path.join(root_dir, h5_filename))
        self.transform = transform
        self.is_test = is_test

    def read_h5(self):
        filename = self.filepath + '.h5'
        with h5py.File(filename, "r") as f:
            key_list = list(f.keys())
            for i, a_group_key in enumerate(key_list):
                if a_group_key=='coordinate':
                    coords = list(f[a_group_key])
                elif a_group_key=='info':
                    info = list(f[a_group_key])
                elif a_group_key=='groundtruth':
                    gt = np.array([])
                else:
                    mono = np.array([])
                break
            #ds_arr = f[a_group_key][()]  # returns as a numpy array
            
        return gt, mono, coords, info

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_dir, 'mono', self.image_list[idx]))
        label = Image.open(os.path.join(self.image_dir, 'groundtruth', self.image_list[idx]))

        if self.transform:
            image, label = self.transform(image, label)
        else:
            image = F.to_tensor(image)
            label = F.to_tensor(label)
        if self.is_test:
            name = self.image_list[idx]
            return image, label, name
        return image, label

    @staticmethod
    def _check_image(lst):
        for x in lst:
            splits = x.split('.')
            if splits[-1] not in ['png', 'jpg', 'jpeg', 'tiff', 'mat', 'h5']:
                raise ValueError

def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    elif ext == '.mat':
        im_dict = scipy.io.loadmat(filename)
        return Image.fromarray(im_dict['blurhyper'])
    else:
        return Image.open(filename)
    
def load_blur_GT_pairs(dir):
    for filename in os.listdir(dir):
        if 'GT' in filename:
            return
        break

def split_im(array, nrows=128, ncols=128):
    """
    Split an image into sub-images. Takes a 3D input array (typically representing an image) and splits it
    into a grid of sub-matrices. The splitting is performed along the x and y axes (dimensions 1 and 2),
    creating nrows x ncols x channels for each sub-image.

    Args:
    - array (numpy.ndarray): The input 3D array to be split. This represents an image tensor.
    - nrows (int): The number of rows to split the image into.
    - ncols (int): The number of columns to split the image into.

    Returns:
    - numpy.ndarray: A 4D numpy array containing the sub-images. The first axis represents 
    the ith sub-image split from left to right and top to down.
    
    Note: This function assumes images of size or near size 512x512 for each channel.
    """
    array = array[:512,:512,:]
    r, h, ch = array.shape
    return (array.reshape(h//nrows, nrows, -1, ncols, ch)
                 .swapaxes(1, 2)
                 .reshape(-1, nrows, ncols, ch))
    

def plot_subimages(array):
    """
    Plot sub-images as a grid of images.

    Args:
    - array (numpy.ndarray): 4D array where array.shape[0] = # of sub-images, 
    array.shape[1] = array.shape[2] = # values along x, y axes, and
    array.shape[2] = # channels.
        
    Returns:
    - None. 
    """
    num_ims = array.shape[0]
    # plot compartments
    fig = plt.figure(figsize=(8, 8))
    for i in range(1, num_ims + 1):
        ax = fig.add_subplot(int(np.ceil(num_ims / int(np.sqrt(num_ims)))), int(np.sqrt(num_ims)), i)
        ax.imshow(array[i-1,:,:,0:3])
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        plt.tight_layout(pad=1)

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

#class h5Dataset(Dataset):