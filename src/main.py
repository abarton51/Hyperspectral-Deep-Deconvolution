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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set deterministic behavior
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)

# Set save and data directories
#datapath = 'I:\Georgia Institute of Technology\Deep Learning Project Group - General'
#savepath = 'I:\Georgia Institute of Technology\Deep Learning Project Group - General\saved_models\DummyNet'
#savepath = 'I:\Georgia Institute of Technology\Deep Learning Project Group - General\saved_models\ClassicUnet'
#------#
# Austin's local directories
datapath = 'C:\\Users\\Teddy\\Documents\\Academics\\Deep Learning\\Projects\\CS_4644_Project\\src\\data'
savepath = 'C:\\Users\\Teddy\\Documents\\Academics\\Deep Learning\\Projects\\CS_4644_Project\\src\\saved_models\\ClassicUnet'
config_file = ''
print(savepath)

# Perform training?
trainingMode = False

# Perform evaluation on pretrained model?
evaluationMode = True

trainLoader, valLoader, testLoader = dataset.getDeblurDataLoader('Dataset', datapath, batch_size=64,
                                                                 split=(0.8, 0.1, 0.1), memload=False)
model = models.ClassicUnet()
#model = models.DummyNet()
numparams = count_params(model)
print(f"Number of parameters: {numparams}")

optimizer = Adam(model.parameters(), lr=0.01)
scheduler = lrScheduler.MultiStepLR(optimizer,milestones=[100,200,300,1000], gamma=0.1)
#scheduler = lrScheduler.StepLR(optimizer, step_size=5, gamma=0.2)

loss_fn = nn.MSELoss()  # to be changed later if needed

trainer = Trainer(trainLoader, valLoader, model, loss_fn, optimizer, savepath, scheduler=None) # Initialize trainer
evaluator = Evaluator(trainer, testLoader) # Initialize Evaluator
trainer.set_evaluator(evaluator) # Set the evaluator inside the trainer

if evaluationMode:
    trainer.load_model(savepath + '/bestmodel.pth')
    loss, psnr = evaluator.test_model('test')
    print(psnr)

if trainingMode:
    starttime = time.time()
    bestmodel,bestepoch,bestloss = trainer.train(10)
    endtime = time.time()

    np.savetxt(savepath + '/' + 'trainloss.txt', trainer.get_train_loss_history(), fmt='%f')
    np.savetxt(savepath + '/' + 'valloss.txt', trainer.get_val_loss_history(), fmt='%f')

    print(f"ELAPSED TIME: {endtime-starttime} SECONDS")
    print(f"Model Parameter Count: {count_params(model)}")
    print(f"Best Epoch: {bestepoch}")
    print(f"Best Loss: {bestloss}")

print('test')
# Plot the loss and accuracy from training history
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
# ax1.plot(epoch_train_loss)
# ax1.set_title("Loss")
# ax2.plot(epoch_train_acc)
# ax2.set_title("Accuracy")
# if (not os.path.exists('../reports')): os.mkdir('../reports')
# plt.savefig('reports/train.png')
