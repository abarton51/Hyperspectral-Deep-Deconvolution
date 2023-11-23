import numpy as np
from matplotlib.colors import LinearSegmentedColormap

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
