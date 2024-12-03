import numpy as np
import torch.nn as nn
import time
import torch
import copy
from models import MyModel
import matplotlib.pyplot as plt
import pathlib
from data_downloader import KittiDataDownloader
from data_processing_kitti import DataProcessorKitti

class SolverKitti(object):
    def __init__(self, **kwargs):

        batch_size = 5

        # Define the data
        self.train_loader, self.val_loader, self.test_dataset = DataProcessorKitti(batch_size)

    def train(self):
        print("Training not yet implemented for KITTI")
