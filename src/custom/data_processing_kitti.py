import torchvision.transforms as transforms
import torchvision
import os
import torch
from torch.utils.data import DataLoader
from data_downloader import KittiDataDownloader

def DataProcessingKitti(batch_size):

    def __init__(self, **kwargs):
        self.downloadDataset()


    def downloadDataset(self):
        kittiDownloader = KittiDataDownloader()
        kittiDownloader.prepareDataset()