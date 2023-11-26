import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def count_params(model):
    """Count the number of parameters in the model. """
    param_count = np.sum([np.prod(p.size()) for p in model.parameters()])
    return param_count

def red_cm():
    colors = np.array([(0, 0, 0), (1, 0, 0)])  # black-red
    cm = LinearSegmentedColormap.from_list(
        "Red", colors, N=256)
    return cm

def blue_cm():
    colors = np.array([(0, 0, 0), (0, 0, 1)])  # black-blue
    cm = LinearSegmentedColormap.from_list(
        "Blue", colors, N=256)
    return cm

def green_cm():
    colors = np.array([(0, 0, 0), (0, 1, 0)])  # black-green
    cm = LinearSegmentedColormap.from_list(
        "Green", colors, N=256)
    return cm

def PSNR(pred, gt):
    loss = nn.MSELoss()
    psnr = 10*np.log10(1/(np.array(loss(gt,pred).to('cpu'))))
    return psnr

def accumulate_grad_flow(named_parameters,grad_flow_ave,grad_flow_max):
    ave_grads = np.array([])
    max_grads = np.array([])
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            ave_grads = np.append(ave_grads,p.grad.abs().mean().cpu())
            max_grads = np.append(max_grads,p.grad.abs().max().cpu())

    if grad_flow_ave is None:
        grad_flow_ave = ave_grads[None,:]
    else:
        grad_flow_ave = np.concatenate((grad_flow_ave, ave_grads[None, :]), 0)

    if grad_flow_max is None:
        grad_flow_max = max_grads[None,:]
    else:
        grad_flow_max = np.concatenate((grad_flow_max,max_grads[None,:]),0)

    return grad_flow_ave, grad_flow_max

def plot_grad_flow(named_parameters,grad_flow_ave,grad_flow_max,saveDirectory,epoch):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)

    max_grads = np.max(grad_flow_max,0)
    ave_grads = np.mean(grad_flow_ave,0)

    plt.figure()
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.5, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.5, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4)], ['max-gradient', 'mean-gradient'])

    #plt.show()
    plt.savefig(saveDirectory + '/figs/' + 'grads' + str(epoch) + '.png', bbox_inches='tight')
    plt.close()