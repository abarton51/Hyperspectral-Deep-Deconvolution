import os

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import models.model as models
from Evaluator import Evaluator
from Trainer import Trainer
import Utils.DataLoader as dataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
datapath = 'I:\Georgia Institute of Technology\Deep Learning Project Group - General'
savepath = 'I:\CS_4644_Project\src\saved_models\DummyNet'
print(savepath)

trainLoader, valLoader, testLoader = dataset.getDeblurDataLoader('Dataset',datapath,batch_size=4,
                                                                 split=(0.01,0.01,0.98),memload=True)

model = models.DummyNet()

optimizer = Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss() # to be changed later if needed

trainer = Trainer(trainLoader, valLoader, model, loss_fn, optimizer, savepath)
trainer.train(1)
print('test')



# Plot the loss and accuracy from training history
#fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
#ax1.plot(epoch_train_loss)
#ax1.set_title("Loss")
#ax2.plot(epoch_train_acc)
#ax2.set_title("Accuracy")
#if (not os.path.exists('../reports')): os.mkdir('../reports')
#plt.savefig('reports/train.png')
