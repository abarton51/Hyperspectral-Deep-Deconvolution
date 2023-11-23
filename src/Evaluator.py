import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt

class Evaluator:
    def __init__(self, dataloader: DataLoader, model: nn.Module, loss_fn: nn.Module):
        self.testdata = dataloader
        self.model = model
        self.criterion = loss_fn

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
