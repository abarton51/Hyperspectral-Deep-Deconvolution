import math

import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import Optimizer


class Trainer:
    def __init__(self, dataLoader: DataLoader, valData: DataLoader, model: nn.Module, loss_fn: nn.Module,
                 optimizer: Optimizer, saveDirectory, config=None):
        self.trainLoader = dataLoader
        self.valLoader = valData
        self.model = model
        self.criterion = loss_fn
        self.optimizer = optimizer
        self.saveDirectory = saveDirectory

        self.train_loss_history = []
        self.val_loss_history = []
        self.bestepoch = 0  # epoch of the currently best model

        # if the user did not present a config, initialize default config
        if config is None:
            config["epochs"] = 100
            config["verbose"] = True
            config["doplot"] = True
            config["saveincrement"] = 25

        self.epochs = config["epochs"]  # total number epochs
        self.verbose = config["verbose"]  # Verbose training status updates?
        self.doplot = config["doplot"]  # output epoch plots?
        self.saveincrement = config["saveincrement"]  # save model weights every 25 epochs

    # initialize a new trainer by loading in the parameters from a model weight file
    # @classmethod
    # def loadModelTrainer(cls, modelname):

    # Single Epoch training loop
    def train_epoch(self):
        self.model.train()
        avgloss = 0;
        for batch, (inputs, gt) in enumerate(self.trainLoader):
            self.optimizer.zero_grad()
            pred = self.model(inputs)
            loss = self.criterion(pred, gt)

            loss.backward()
            self.optimizer.step()

            # update avg loss of this epoch
            avgloss += loss

            # Verbose progress monitor
            if self.verbose:
                if batch % 100 == 0:
                    print(f"batch {batch + 1} loss: {loss:>7f}")

        # record loss history
        avgloss = avgloss / (batch + 1)
        self.train_loss_history.append(avgloss)
        return avgloss

    # Training the model over multiple epochs
    def train(self, epochs=None):

        if epochs is None:
            epochs = self.epochs

        bestloss = math.inf
        for epoch in range(epochs):
            print("------------------------------------------------")
            print(f"Epoch {epoch + 1}:")
            trainloss = self.train_epoch()
            print(f"Training Loss {trainloss}:")

            vloss = self.validate()
            print(f"Validation Loss {vloss}:")

            # Save the best model
            if vloss < bestloss:
                self.bestepoch = epoch
                self.save_model('bestmodel.pth')

            # Save model every "saveincrement" epochs
            if (epoch+1) % self.saveincrement == 0:
                self.save_model('epoch' + str(epoch) + '.pth')

        print("Training Done!")

    # Validation against validation dataset
    def validate(self):
        self.model.eval()
        avgloss = 0

        # no grad for validation
        with torch.no_grad():
            for i, (inputs, gt) in enumerate(self.valLoader):
                pred = self.model(inputs)
                loss = self.criterion(pred, gt)

                # update avg validation loss
                avgloss += loss

            avgloss = avgloss / (i + 1)
            self.val_loss_history.append(avgloss)
            return avgloss

    # Save the current model parameters
    def save_model(self, fname):
        outPath = self.saveDirectory + '/' + fname
        torch.save(self.model.state_dict(), outPath)

    # Load a model parameters into the trainer model
    def load_model(self, filedir):
        self.model.load_state_dict(torch.load(filedir))
        self.model.eval()


def evaluate(dataloader: DataLoader, model: nn.Module, loss_fn: nn.Module):
    model.eval()
    num_batches = len(dataloader)
    test_loss = 0
    with torch.no_grad():  # to make sure no gradient is calculated
        for input, label in dataloader:
            pred = model(input)
            test_loss += loss_fn(pred, label).item()

        test_loss /= num_batches
        print(f"Avg loss: {test_loss:>8f} \n")
