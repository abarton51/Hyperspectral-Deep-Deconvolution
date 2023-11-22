import os
from models.model import build_unet
from utils import train_loop, test_loop

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

print(torch.cuda.is_available())

train_dataloader = DataLoader() # to be changed later to the actual dataloader
test_dataloader = DataLoader()

model = classic_unet()
optimizer = Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss() # to be changed later if needed

epochs = 10
epoch_train_loss = []
epoch_train_acc = []
for epoch in range (epochs):
    print("------------------------------------------------")
    print(f"Epoch {epoch+1}:")
    train_loss_history, train_acc_history = train_loop(train_dataloader, model, loss_fn, optimizer)
    epoch_train_loss.append(train_loss_history[-1]) # could use some form of aggregation rather than just take the last value
    epoch_train_acc.append(train_acc_history[-1])
    test_loop(test_dataloader, model, loss_fn)
print("Done!")

# Plot the loss and accuracy from training history
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
ax1.plot(epoch_train_loss)
ax1.set_title("Loss")
ax2.plot(epoch_train_acc)
ax2.set_title("Accuracy")
if (not os.path.exists('../reports')): os.mkdir('../reports')
plt.savefig('reports/train.png')
