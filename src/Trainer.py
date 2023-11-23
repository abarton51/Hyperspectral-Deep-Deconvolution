import math
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import Optimizer
from tqdm import tqdm
import numpy as np

class Trainer:
    def __init__(self, dataLoader: DataLoader, valData: DataLoader, model: nn.Module, loss_fn: nn.Module,
                 optimizer: Optimizer, saveDirectory, scheduler=None, config=None):
        self.trainLoader = dataLoader
        self.valLoader = valData
        self.evaluator = None

        self.criterion = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.saveDirectory = saveDirectory

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(self.device)
        self.model = model.to(self.device)

        self.train_loss_history = np.array([],dtype=np.float32)
        self.val_loss_history = np.array([],dtype=np.float32)

        # if the user did not present a co initialize default config
        if config is None:
            config = dict()
            config["epochs"] = 1000
            config["doplot"] = True
            config["saveincrement"] = 2

        self.epochs = config["epochs"]  # total number epochs
        self.doplot = config["doplot"]  # output epoch plots?
        self.saveincrement = config["saveincrement"]  # save model weights every few epochs

    # initialize a new trainer by loading in the parameters from a model weight file
    # @classmethod
    # def loadModelTrainer(cls, modelname):

    # Single Epoch training loop
    def train_epoch(self):
        self.model.train()
        avgloss = torch.Tensor([0])
        with tqdm(self.trainLoader, position = 0, desc = 'Batch') as tqdm_data_loader:
            for batch, (inputs, gt, _) in enumerate(tqdm_data_loader):
                self.optimizer.zero_grad()

                # Put data on target device
                inputs = inputs.to(self.device)
                gt = gt.to(self.device)

                #get loss & backprop
                pred = self.model(inputs)
                loss = self.criterion(pred, gt)
                loss.backward()

                #step gradients
                self.optimizer.step()

                # update avg loss of this epoch
                avgloss += loss.detach().to('cpu')

        # record loss history
        avgloss = avgloss / (batch + 1)
        avgloss = np.array(avgloss)
        self.train_loss_history = np.append(self.train_loss_history,avgloss)
        return avgloss[0]

    # Training the model over multiple epochs
    def train(self, epochs=None):

        if epochs is None:
            epochs = self.epochs

        #track which epoch had the lowest validation loss
        bestepoch = 0
        bestloss = math.inf

        print("------------------------------------------------")
        print("Training Start")
        print("------------------------------------------------")

        with tqdm(range(epochs), position=1, initial=1, desc='Training Progress') as tqdm_epochs:
            for epoch in tqdm_epochs:

                # Train one epoch and validate
                trainloss = self.train_epoch()
                vloss = self.validate()

                # If we do want learning rate decay, step it now
                if self.scheduler is not None:
                    self.scheduler.step()

                # Save the best model
                if vloss < bestloss:
                    bestepoch = epoch
                    bestloss = vloss
                    self.save_model('bestmodel.pth')

                # Save model every "saveincrement" epochs
                # Also perform evaluation
                if (epoch + 1) % self.saveincrement == 0:
                    self.save_model('epoch' + str(epoch + 1) + '.pth')
                    if self.evaluator is not None:
                        self.evaluator.test_model(str(epoch + 1))

                tqdm_epochs.set_postfix({"Epoch": epoch, "Train Loss": trainloss, "Val Loss": vloss})


        print("------------------------------------------------")
        print("Training Done!")
        print("------------------------------------------------")

        # Return the most successful model
        self.load_model(self.saveDirectory + '/' + 'bestmodel.pth')
        return self.model,bestepoch,bestloss


    # Validation against validation dataset
    def validate(self):
        self.model.eval()
        avgloss = torch.Tensor([0])

        # no grad for validation
        with torch.no_grad():
            for i, (inputs, gt, _) in enumerate(self.valLoader):

                # data to target device
                inputs = inputs.to(self.device)
                gt = gt.to(self.device)

                # validation loss
                pred = self.model(inputs)
                loss = self.criterion(pred, gt)

                # update avg validation loss
                avgloss += loss.detach().to('cpu')

            avgloss = avgloss / (i + 1)
            avgloss = np.array(avgloss)
            self.val_loss_history = np.append(self.val_loss_history, avgloss)
            return avgloss[0]

    # Save the current model parameters
    def save_model(self, fname):
        outPath = self.saveDirectory + '/' + fname
        torch.save(self.model.state_dict(), outPath)

    # Load a model parameters into the trainer model
    def load_model(self, filedir):
        self.model.load_state_dict(torch.load(filedir, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()

    def get_train_loss_history(self):
        return self.train_loss_history

    def get_val_loss_history(self):
        return self.val_loss_history

    def set_evaluator(self, evaluator):
        self.evaluator = evaluator

    def get_model(self):
        return self.model

    def get_criterion(self):
        return self.criterion

    def get_save_dir(self):
        return self.saveDirectory