# from data_loaders.cifar10 import CIFAR10 # data loader for cifar10
import torchvision.transforms as transforms
import torchvision
import os
import torch
from torch.utils.data import DataLoader

def DataProcessing(batch_size):
    '''
    Loads the image data and processes it with the dataloader files 
    
    Args:
        batch_size
        
    Returns:
        train_loader: training data
        val_loader: validation data
        test_dataset: testing data
        
    '''
    
    #Tranformations for training data
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
        ]
    )
    
    # Transforms for validation data (no augmentation)
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
        ]
    )

    # Load the training dataset
    train_dataset = torchvision.datasets.CIFAR10(
            root=os.path.join('.', "data", "cifar10"),
            train=True,
            download=True,
            transform=transform_train,
        )

    # Wrap training dataset with batchsize and shuffling
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    # Load the testing dataset
    test_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )
    
    # Wrap testing dataset with batchsize and shuffling for validation
    val_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=100, shuffle=False, num_workers=2
    )
    
    return train_loader, val_loader, test_dataset