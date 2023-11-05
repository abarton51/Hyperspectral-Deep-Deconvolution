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

def h5_dataloader(path, batch_size=64, num_workers=0, use_transform=False):
    #image_dir = os.path.join(path, 'Dataset.h5')
    image_dir = path
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
    #image_dir = os.path.join(path, 'train')
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

def valid_dataloader(path, batch_size=64, num_workers=0, use_transform=True):
    #image_dir = os.path.join(path, 'train')
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

class DeblurDataset(Dataset):
    def __init__(self, h5_filename='Dataset', root_dir='data', blur=True, transform=None, is_test=False):
        self.root_dir = root_dir
        self.filepath = os.path.join(root_dir, h5_filename)
        self.transform = transform
        self.is_test = is_test
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
    
'''
def _load_h5_dataset(self, file_name, mono=True):
    """Method for loading .h5 files
    
        returns: dict that contains name of the .h5 file as stored in the .h5 file, as well as a generator of the data
    """
    path = os.path.join(self.dir_path, file_name)
    file = h5py.File(path)
    if mono:
        key = 'mono'
    else:
        key = 'groundtruth'
    data = file[key]
    #return dict(file=file, data=data)
    return data
'''
