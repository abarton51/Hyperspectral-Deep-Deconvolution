import numpy as np
import pandas as pd

import os

from PIL import Image
from PIL import ImageShow
import spectral as spec
import tarfile
import scipy.io
from scipy.interpolate import interp1d

import torch

from torch import nn
from torch.optim import Adam
import torch.optim.lr_scheduler as lrScheduler

from matplotlib import pyplot as plt
import seaborn as sn
import tqdm

from Utils.Imports import *
from Utils import DataLoader
from Utils import DeblurLoss
from Evaluator import Evaluator
from Trainer import Trainer
import torchvision as ttv
import models.model as models

print(torch.__version__)
device = torch.device(GetLowestGPU(pick_from=[0,1,2,3]))
# helper functions
def to_torch(x):
    try:
        return torch.from_numpy(x).float().to(device)
    except:
        return x
def to_numpy(x):
    return x.detach().cpu().numpy()
    
datapath = 'C:\\Users\\Teddy\\Documents\\Academics\\Deep Learning\\Projects\\CS_4644_Project\\src\\data'
save_path = 'C:\\Users\\Teddy\\Documents\\Academics\\Deep Learning\\Projects\\CS_4644_Project\\src\\saved_models\\ClassicUnet'

trainLoader, valLoader, testLoader = DataLoader.getDeblurDataLoader('Dataset', datapath, batch_size=64,
                                                                 split=(0.8, 0.1, 0.1), memload=False)
model = models.ClassicUnet()

optimizer = Adam(model.parameters(), lr=0.01)

loss_fn = nn.MSELoss()
#loss_fn = DeblurLoss.DeblurCustomLoss(window_size=7, size_average=True, is_loss=True, plus_mse=True, mse_weight=0.5, ssim_weight=0.25)

# Because how the code has been set up, the Evalutor needs a Trainer. So we make a default Trainer so we can use the Evaluator (WE ARE NOT TRAINING)
trainer = Trainer(trainLoader, valLoader, model, loss_fn, optimizer, save_path, scheduler=None)
evaluator = Evaluator(trainer, testLoader) # Initialize Evaluator
trainer.set_evaluator(evaluator) # Set the evaluator inside the trainer

trainer.load_model(save_path + '/bestmodel.pth')
modelt = trainer.get_model()
#loss, psnr = evaluator.test_model('test')

filename = "src/data/InterestingDataset.h5"

def read_h5(filename, num_samples):
    h5 = h5py.File(filename,'r')
    gt_data = h5['groundtruth']
    mono_data = h5['mono']
    gts = torch.tensor([])
    inputs = []
    preds = torch.tensor([])
    for i in range(num_samples):
        gt_tensor = torch.tensor(gt_data[i][None,:,:])
        gts = torch.cat([gts, torch.permute(gt_tensor, (0, 2, 3, 1))], dim=0)
        inputs.append(torch.tensor(mono_data[i][None,None,:,:]))
        pred_i = to_torch(modelt(inputs[i])[0])[None,:,:,:]
        # C, H, W --> H, W, C
        preds = torch.cat([preds, torch.permute(pred_i, (0, 2, 3, 1))], dim=0)
    h5.close()
    return gts, inputs, preds

def pixel_compare(gt, pred, save_path, coords=None, pic_names=None, seed=None):
    if seed!=None:
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
    
    channels = np.arange(gt.shape[3])
    num_samples = gt.shape[0]
    if coords==None:
        r1 = torch.randint(0, 128, size=(2, ))
        r2 = torch.randint(0, 128, size=(2, ))
        r3 = torch.randint(0, 128, size=(2, ))
        coords = [r1, r2, r3]

    if pic_names==None:
        pic_names = ['Image'] * num_samples

    plt.style.use('seaborn')
    marker_colors = ['b', 'm', 'r']
    gt_line_colors = ['dodgerblue', 'blueviolet', 'firebrick']
    pred_line_colors = ['darkturquoise', 'orchid', 'tomato']
    c_len = len(marker_colors)
    for j in range(num_samples):
        prgb_gt_j = gt[j,:,:,(28, 14, 0)]
        prgb_pred_j = pred[j,:,:,(28, 14, 0)]
        fig, ax = plt.subplots()
        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()
        for i, ri in enumerate(coords):
            gt_ri = to_numpy(gt[j, ri[0], ri[1]])
            pred_ri = to_numpy(pred[j, ri[0], ri[1]])

            c_idx = i % c_len
            c_marker = marker_colors[c_idx]
            c_gt_line = gt_line_colors[c_idx]
            c_pred_line = pred_line_colors[c_idx]

            rand_pixel_str = 'Random Pixel ' + str(i+1)

            ax.scatter(channels, pred_ri, s=15, c=c_marker, marker='v')
            ax.scatter(channels, gt_ri, s=15, c=c_marker)
            ax.plot(gt_ri, c=c_gt_line, label='Ground Truth ' + rand_pixel_str)
            ax.plot(pred_ri, c=c_pred_line, label='Predicted ' + rand_pixel_str)
            ax.set_xlabel('Channels; 420-700 nm')
            ax.set_ylabel('Raw Pixel Values')
            ax.set_title(pic_names[j] + '; Pixel Value Comparison across Wavelength')
            ax.legend()
            if seed!=None:
                fig.savefig(save_path + '\\' + str(seed) + '_seed_' + pic_names[j] + '_pixel_plot')
            else:
                fig.savefig(save_path + '\\' + pic_names[j] + '_pixel_plot')

            ax1.imshow(prgb_gt_j)
            ax1.plot(ri[0], ri[1], marker='v', color=c_marker, markersize=10, label=rand_pixel_str)
            l1 = ax1.legend()
            for text in l1.get_texts():
                text.set_color("white")
            ax2.imshow(to_numpy(prgb_pred_j))
            ax2.plot(ri[0], ri[1], marker='v', color=c_marker, markersize=10, label=rand_pixel_str)
            l2 = ax2.legend()
            for text in l2.get_texts():
                text.set_color("white")

        ax1.set_title('Ground Truth')
        ax2.set_title('Predicted')
        if seed!=None:
            fig1.savefig(save_path + '\\' + str(seed) + '_seed_' + pic_names[j] + '_gt_markedpixels')
            fig2.savefig(save_path + '\\' + str(seed) + '_seed_'  + pic_names[j] + '_pred_markedpixels')
        else:
            fig1.savefig(save_path + '\\' + pic_names[j] + '_gt_markedpixels')
            fig2.savefig(save_path + '\\' + pic_names[j] + '_pred_markedpixels')


seed=None # set to None if using manually defined coordinates
coords = [[120, 80], [40, 20], [64, 42]] # set to None if using randomly generated coordiates

gt_data, inputs, preds = read_h5(filename, 4)
pic_names = ['Beads', 'Doll', 'Flowers', 'Public_Space']
save_plot_path = save_path + '\\figs\\pixel_comparison'

pixel_compare(gt_data, preds, save_plot_path, coords=coords, pic_names=pic_names, seed=seed)