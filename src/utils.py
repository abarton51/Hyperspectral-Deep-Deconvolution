import math

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import Optimizer


class Trainer:
    def __init__(self, dataLoader: DataLoader, valData: DataLoader, model: nn.Module, loss_fn: nn.Module,
                 optimizer: Optimizer, saveDirectory, config = None):
        self.trainLoader = dataLoader
        self.valLoader = valData
        self.model = model
        self.criterion = loss_fn
        self.optimizer = optimizer
        self.saveDirectory = saveDirectory


        self.train_loss_history = []
        self.val_loss_history = []
        self.bestepoch = 0  # epoch of the currently best model

        #if the user did not present a config, initialize default config
        if config is None:
            config["epochs"] = 100
            config["verbose"] = True
            config["doplot"] = True
            config["saveincrement"] = 25

        self.epochs = config["epochs"] #total number epochs
        self.verbose = config["verbose"] #Verbose training status updates?
        self.doplot = config["doplot"] #output epoch plots?
        self.saveincrement = config["saveincrement"] #save model weights every 25 epochs

    def train_epoch(self):
        self.model.train()
        avgloss = 0;
        for batch, (inputs, gt) in enumerate(self.trainLoader):
            self.optimizer.zero_grad()
            pred = self.model(inputs)
            loss = self.criterion(pred, gt)


            loss.backward()
            self.optimizer.step()

            #update avg loss of this epoch
            avgloss += loss

            #Verbose progress monitor
            if self.verbose:
                if batch % 100 == 0:
                    print(f"batch {batch + 1} loss: {loss:>7f}")

        # record loss history
        avgloss = avgloss/(batch+1)
        self.train_loss_history.append(avgloss)
        return avgloss

    def train(self)

        bestloss = math.inf
        for epoch in range(self.epochs):
            print("------------------------------------------------")
            print(f"Epoch {epoch + 1}:")
            trainloss = self.train_epoch()
            print(f"Training Loss {trainloss}:")

            vloss = self.validate()
            print(f"Validation Loss {vloss}:")

            #Save the best model
            if vloss < bestloss:
                self.bestepoch = epoch


        print("Done!")


    def validate(self):
        self.model.eval()
        avgloss = 0
        with torch.no_grad():
            for i, (inputs, gt) in enumerate(self.valLoader):
                pred = self.model(inputs)
                loss = self.criterion(pred, gt)

                # update avg validation loss
                avgloss += loss

        avgloss = avgloss/(i+1)
        self.val_loss_history.append(avgloss)
        return avgloss



def test_loop(dataloader: DataLoader, model: nn.Module, loss_fn: nn.Module):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    with torch.no_grad():  # to make sure no gradient is calculated
        for input, label in dataloader:
            # this code was mostly copied from the pytorch tutorial. I don't know why they didn't use enumerate or if num_batches is different from size.
            pred = model(input)
            test_loss += loss_fn(pred, label).item()
            correct += (pred.argmax(1) == label).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
