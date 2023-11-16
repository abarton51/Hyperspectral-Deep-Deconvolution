import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import Optimizer

def train_loop(dataloader: DataLoader, model: nn.Module, loss_fn: nn.Module, optimizer: Optimizer):
    model.train()
    train_loss_history = []
    train_acc_history = []
    for batch, (input, label) in enumerate(dataloader):
        pred = model(input)
        loss = loss_fn(pred, label)

        # record history
        train_loss_history.append(loss)
        train_acc_history.append(torch.isclose(pred, label, rtol=0).type(torch.float).mean())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            print(f"batch {batch+1} loss: {loss:>7f}")
    
    return train_loss_history, train_acc_history

def test_loop(dataloader: DataLoader, model: nn.Module, loss_fn: nn.Module):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    with torch.no_grad(): # to make sure no gradient is calculated
        for input, label in dataloader:
            # this code was mostly copied from the pytorch tutorial. I don't know why they didn't use enumerate or if num_batches is different from size.
            pred = model(input)
            test_loss += loss_fn(pred, label).item()
            correct += (pred.argmax(1) == label).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
