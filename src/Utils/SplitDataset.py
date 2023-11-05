from Utils.Imports import *
from Utils.DataLoader import read_h5

class MyData:
    def __init__(self, file_path):
        self.file_path = file_path
        self.datasets = {}
        self.mono_ds = None
        self.GT_ds = None
        self.coords = None
        self.info = None
        
    def read_data(self):
        self.GT_ds, self.mono_ds, self.coords, self.info = read_h5(self.file_path)

def split(GT_ds, mono_ds, coords, info):
    return GT_ds.__getitem__(0)