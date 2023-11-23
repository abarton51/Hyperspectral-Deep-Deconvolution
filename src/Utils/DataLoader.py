from Utils.Imports import *

from os import listdir
from os.path import splitext, isfile, join
from Utils.DataAug import PairCompose, PairRandomCrop, PairRandomHorizontalFilp, PairToTensor
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
import torch.utils.data
import torch

device = torch.device(GetLowestGPU(pick_from=[0, 1, 2, 3]))


def to_torch(x):
    return torch.from_numpy(x).float().to(device)


def to_numpy(x):
    return x.detach().cpu().numpy()


def getDeblurDataLoader(filename, path, batch_size=64, split=(0.8, 0.1, 0.1), num_workers=0, use_transform=False, seed=1):
    dataset = DeblurDataset(filename, root_dir=path)

    generator = torch.Generator().manual_seed(seed)

    trainset, valset, testset = torch.utils.data.random_split(dataset, split, generator=generator)
    trainLoader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    valLoader = DataLoader(
        valset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    testLoader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    return trainLoader, valLoader, testLoader


class DeblurDataset(Dataset):
    def __init__(self, h5_filename='Dataset', root_dir='data', transform=None):
        self.root_dir = root_dir
        self.filepath = os.path.join(root_dir, h5_filename)
        self.transform = transform
        filename = h5_filename + '.h5'
        self.h5_file = h5py.File(os.path.join(root_dir, filename))
        self.coords = self.h5_file.get('coords')
        self.mono_ds = self.h5_file.get('mono')
        self.gt_ds = self.h5_file.get('groundtruth')
        self.info = self.h5_file.get('info')

    def __len__(self):
        return self.info.shape[0]

    def __getitem__(self, idx):
        mono_image = self.mono_ds.__getitem__(idx)
        gt_image = self.gt_ds.__getitem__(idx)

        if self.transform:
            mono_image, gt_image = self.transform(mono_image, gt_image)
        else:
            mono_image = torch.Tensor(mono_image)

            # In case that we are reading a single-channel image (which we are), add singleton dim
            # This is laying framework in case multiple focal image channels are needed
            if len(mono_image.shape) != 3:
                mono_image = mono_image[None,:,:]
            gt_image = torch.Tensor(gt_image)

        return mono_image, gt_image

    def __del__(self):
        self.h5_file.close()