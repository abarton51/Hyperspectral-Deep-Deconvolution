from Utils.Imports import *

from os import listdir
from os.path import splitext, isfile, join
from Utils.DataAug import PairCompose, PairRandomCrop, PairRandomHorizontalFilp, PairToTensor
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader

device = torch.device(GetLowestGPU(pick_from=[0,1,2,3]))
def to_torch(x):
    return torch.from_numpy(x).float().to(device)
def to_numpy(x):
    return x.detach().cpu().numpy()

def train_dataloader(path, batch_size=64, num_workers=0, use_transform=True):
    transform = None
    if use_transform:
        transform = PairCompose(
            [
                PairRandomHorizontalFilp(), #PairRandomCrop(128)
                PairToTensor()
            ]
        )
    dataloader = DataLoader(
        DeblurDataset(path, transform=transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader

def test_dataloader(path, batch_size=1, num_workers=0):
    dataloader = DataLoader(
        DeblurDataset(path),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader

def valid_dataloader(path, batch_size=64, num_workers=0, use_transform=True):
    transform = None
    if use_transform:
        transform = PairCompose(
            [
                PairRandomHorizontalFilp(),
                PairToTensor()
            ]
        )
    dataloader = DataLoader(
        DeblurDataset(path, transform=transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader

class DeblurDataset(Dataset):
    def __init__(self, h5_filename='Dataset', root_dir='data', transform=None):
        self.root_dir = root_dir
        self.filepath = os.path.join(root_dir, h5_filename)
        self.transform = transform
        filename = h5_filename + '.h5'
        self.gt_ds, self.mono_ds, self.coords, self.info = read_h5(filename)

    def __len__(self):
        return self.info.shape[0]

    def __getitem__(self, idx):
        mono_image = self.mono_ds.__getitem__(idx)
        gt_image = self.gt_ds.__getitem__(idx)

        if self.transform:
            mono_image, gt_image = self.transform(mono_image, gt_image)
        else:
            image = F.to_tensor(mono_image)
            label = F.to_tensor(gt_image)
        if self.is_test:
            name = self.info[idx]
            return image, label, name
        return image, label
    
    def __del__(self):
        self.h5_file.close()

    @staticmethod
    def _check_image(lst):
        for x in lst:
            splits = x.split('.')
            if splits[-1] not in ['png', 'jpg', 'jpeg', 'tiff', 'mat', 'h5']:
                raise ValueError

def read_h5(filepath):
        filename = filepath + '.h5'
        with h5py.File(filename, "r") as f:
            key_list = list(f.keys())
            for i, a_group_key in enumerate(key_list):
                if a_group_key=='coordinate':
                    coords = list(f[a_group_key])
                elif a_group_key=='info':
                    info = list(f[a_group_key])
                elif a_group_key=='groundtruth':
                    gt_ds = f[a_group_key]
                else:
                    mono_ds = f[a_group_key]
        return gt_ds, mono_ds, coords, info