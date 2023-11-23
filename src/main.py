import os

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

import models.model as models
from Evaluator import Evaluator
from Trainer import Trainer
import Utils.DataLoader as dataset
from Utils.utils import count_params

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
datapath = 'I:\Georgia Institute of Technology\Deep Learning Project Group - General'
#savepath = 'I:\CS_4644_Project\src\saved_models\DummyNet'
savepath = 'I:\CS_4644_Project\src\saved_models\ClassicUnet'
print(savepath)

trainLoader, valLoader, testLoader = dataset.getDeblurDataLoader('Dataset', datapath, batch_size=64,
                                                                 split=(0.8, 0.1, 0.1), memload=True)

model = models.ClassicUnet()
numparams = count_params(model)
print(f"Number of parameters: {numparams}")

optimizer = Adam(model.parameters(), lr=0.01)
scheduler = StepLR(optimizer, step_size=100, gamma=0.5)
#scheduler = StepLR(optimizer, step_size=5, gamma=0.2)

loss_fn = nn.MSELoss()  # to be changed later if needed

trainer = Trainer(trainLoader, valLoader, model, loss_fn, optimizer, savepath, scheduler=None)

bestmodel,bestepoch,bestloss = trainer.train(1000)

print(count_params(model))
print(f"Best Epoch: {bestepoch}")
print(f"Best Loss: {bestloss}")

# Plot the loss and accuracy from training history
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
# ax1.plot(epoch_train_loss)
# ax1.set_title("Loss")
# ax2.plot(epoch_train_acc)
# ax2.set_title("Accuracy")
# if (not os.path.exists('../reports')): os.mkdir('../reports')
# plt.savefig('reports/train.png')
