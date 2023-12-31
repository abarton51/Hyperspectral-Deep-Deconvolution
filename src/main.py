import os
import time
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.optim.lr_scheduler as lrScheduler
import matplotlib.pyplot as plt
import numpy as np
import random

import models.model as models
from Evaluator import Evaluator
from Trainer import Trainer
import Utils.DataLoader as dataset
from Utils.utils import count_params

from Utils import DeblurLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set deterministic behavior
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)

# Set save and data directories
datapath = 'I:\Georgia Institute of Technology\Deep Learning Project Group - General'
#savepath = 'I:\Georgia Institute of Technology\Deep Learning Project Group - General\saved_models\DummyNet'
savepath = 'I:\Georgia Institute of Technology\Deep Learning Project Group - General\saved_models\ReducedUnet'
#----------------------------#
# Austin's local directories
#datapath = 'C:\\Users\\Teddy\\Documents\\Academics\\Deep Learning\\Projects\\CS_4644_Project\\src\\data'
#savepath = 'C:\\Users\\Teddy\\Documents\\Academics\\Deep Learning\\Projects\\CS_4644_Project\\src\\saved_models\\ClassicUnet'
#savepath = 'C:\\Users\\Teddy\\Documents\\Academics\\Deep Learning\\Projects\\CS_4644_Project\\src\\saved_models\\DummyNet'


print(savepath)

# Perform training?
trainingMode = False

# Perform evaluation on pretrained model?
evaluationMode = True

print('Begin Dataset loading...')
trainLoader, valLoader, testLoader = dataset.getDeblurDataLoader('Dataset', datapath, batch_size=128,
                                                                 split=(0.8, 0.1, 0.1), memload=True)

print('Dataset loaded')
model = models.ReducedUnet()
numparams = count_params(model)
print(f"Number of parameters: {numparams}")

optimizer = Adam(model.parameters(), lr=0.1)
scheduler = lrScheduler.MultiStepLR(optimizer,milestones=[200,600], gamma=0.1)
#scheduler = lrScheduler.StepLR(optimizer, step_size=5, gamma=0.2)

#loss_fn = nn.MSELoss()  # to be changed later if needed
# Important Note: SSIM loss is in [0, 2]. Therefore, we need to set the MSE and SSIM weight to get desired loss behavior
loss_fn = DeblurLoss.DeblurCustomLoss(window_size=7, size_average=True, is_loss=True, plus_mse=True, mse_weight=0, ssim_weight=1)

trainer = Trainer(trainLoader, valLoader, model, loss_fn, optimizer, savepath, scheduler=scheduler) # Initialize trainer
evaluator = Evaluator(trainer, testLoader) # Initialize Evaluator
trainer.set_evaluator(evaluator) # Set the evaluator inside the trainer



if trainingMode:
    starttime = time.time()
    bestmodel,bestepoch,bestloss = trainer.train(1000)
    endtime = time.time()

    np.savetxt(savepath + '/' + 'trainloss.txt', trainer.get_train_loss_history(), fmt='%f')
    np.savetxt(savepath + '/' + 'valloss.txt', trainer.get_val_loss_history(), fmt='%f')


    print(f"ELAPSED TIME: {endtime-starttime} SECONDS")
    print(f"Model Parameter Count: {count_params(model)}")
    print(f"Best Epoch: {bestepoch}")
    print(f"Best Loss: {bestloss}")

if evaluationMode:
    trainer.load_model(savepath + '/bestmodel.pth')
    loss, psnr, basepsnr = evaluator.test_model('test')
    np.savetxt(savepath + '/' + 'testlossSSIM.txt', evaluator.get_test_loss_history(), fmt='%f')
    #np.savetxt(savepath + '/' + 'psnr.txt', evaluator.get_psnr_history(), fmt='%f')
    #np.savetxt(savepath + '/' + 'baselinepsnr.txt', evaluator.get_base_psnr_history(), fmt='%f')
    print(psnr)