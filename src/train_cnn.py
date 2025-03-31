import numpy as np
import matplotlib.pyplot as plt
import mlflow
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

from .cnn import ConvNet

def load_cifar_dataset(data_path: Path,
                       batch_size: int = 64,
                       num_workers: int = 2,
                       data_transform: transforms.Compose = None) -> tuple:
    """
    Load CIFAR-10 dataset, return train and test dataloaders

    Args:
        data_path (Path): Path where dataset is stored
        batch_size (int): Batch size for dataloaders
        num_workers (int): Number of workers for dataloaders
        data_transform (transforms.Compose): Data transformation to apply
    Returns:
        Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]: train and test dataloaders
    """
    # Create Train Dataloader
    trainset = torchvision.datasets.CIFAR10(root=data_path, train=True,
                                        download=True, transform=data_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)
    
    # Create Test Dataloader
    testset = torchvision.datasets.CIFAR10(root=data_path, train=False,
                                        download=True, transform=data_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)
    
    return trainloader, testloader

def visualize_batch(dataloader):
    """
    Visualise a batch of images from the training set

    Args:
        train_dataloader (torch.utils.data.DataLoader): Dataloader for training set
    """
    dataiter = iter(dataloader)
    images, _ = next(dataiter)

    # Show images
    img = torchvision.utils.make_grid(images[:4])
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig('../results/visualize_batch.png')

if __name__ == '__main__':
    """
    This script loads the CIFAR-10 dataset and runs classification using a standard CNN.

    The results and model are logged using MLFlow
    """

    # Apply data transformations
    # CIFAR-10 is in range [0,1], normalise to [-1,1]
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load CIFAR-10 dataset
    data_path = Path('../data')
    batch_size = 64
    num_workers = 2
    print('Loading CIFAR-10 dataset...')
    trainloader, testloader = load_cifar_dataset(data_path, batch_size, num_workers, data_transform)

    # Visualise a batch of images (optional)
    visualize_batch = False
    if visualize_batch:
        print('Visualising a batch of images from training set...')
        visualize_batch(trainloader)

    # Create mode
    num_filters_in = 6
    num_filters_out = 16
    kernel = 5
    n_linear_first = 120
    n_linear_second = 84

    cnn_model = ConvNet(num_filters_in=num_filters_in,
                        num_filters_out=num_filters_out,
                        kernel=kernel,
                        n_linear_first=n_linear_first,
                        n_linear_second=n_linear_second)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cnn_model.to(device)
    print(cnn_model)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn_model.parameters(), lr=0.001)





    




