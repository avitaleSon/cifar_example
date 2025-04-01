import numpy as np
import matplotlib.pyplot as plt
import mlflow
from pathlib import Path
import shutil

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler

from cnn import ConvNet

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def load_cifar_dataset(data_path: Path,
                       batch_size: int = 64,
                       num_workers: int = 2,
                       validation: float = 0.1,
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
    # Create Train & Validation Dataloader
    trainset = torchvision.datasets.CIFAR10(root=data_path, train=True,
                                        download=True, transform=data_transform)
    validset = torchvision.datasets.CIFAR10(root=data_path, train=True,
                                        download=True, transform=data_transform)
    num_train = len(trainset)
    indices = list(range(num_train))
    split = int(np.floor(validation * num_train))
    np.random.shuffle(indices)
    train_idx, valid_idx = indices[split:], indices[:split]

    # Create samplers for training and validation sets
    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idx)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,sampler=train_sampler,num_workers=2)
    validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size,
                                            sampler=valid_sampler,num_workers=2)
    # Create Test Dataloader
    testset = torchvision.datasets.CIFAR10(root=data_path, train=False,
                                        download=True, transform=data_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)
    
    return trainloader, validloader, testloader

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
    valid_size = 0.2
    print('Loading CIFAR-10 dataset...')
    trainloader, validloader, testloader = load_cifar_dataset(data_path, 
                                                              batch_size,
                                                              num_workers,
                                                              validation=valid_size,
                                                              data_transform=data_transform)

    # Visualise a batch of images (optional)
    visualize_batch = False
    if visualize_batch:
        print('Visualising a batch of images from training set...')
        visualize_batch(trainloader)

    # Create model
    num_filters_in = 6
    num_filters_out = 16
    kernel = 5
    n_linear_first = 120
    n_linear_second = 84
    batch_norm = True
    cnn_model = ConvNet(num_filters_in=num_filters_in,
                        num_filters_out=num_filters_out,
                        kernel=kernel,
                        n_linear_first=n_linear_first,
                        n_linear_second=n_linear_second,
                        batch_norm=batch_norm)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cnn_model.to(device)
    print(cnn_model)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn_model.parameters(), lr=0.001)

    # Training Loop
    print('Enter Training Loop...')
    num_epochs = 5
    train_loss, valid_loss = [], []

    best_train_loss = float('inf')
    best_valid_loss = float('inf')

    for i_epoch in range(num_epochs):
        # Run Training for each epoch
        for i, data in enumerate(trainloader,0):
            # Get the inputs and labels
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = cnn_model(inputs)
            loss_value = criterion(outputs, labels)

            # Backward pass and optimize
            loss_value.backward()
            optimizer.step()

            # Print statistics
            if i % 100 == 0:
                print(f'Training: Epoch [{i_epoch + 1}/{num_epochs}], Step [{i + 1}/{len(trainloader)}], Loss: {loss_value.item():.4f}')
            train_loss.append(loss_value.item())
            if loss_value.item() < best_train_loss:
                print('Found new best training loss')
                best_train_loss = loss_value.item()

        # Run Validation for each epoch
        for i, data in enumerate(validloader, 0):
            # Get the inputs and labels
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = cnn_model(inputs)
            loss_value = criterion(outputs, labels)

            # Print statistics
            if i % 100 == 0:
                print(f'Validation: Epoch [{i_epoch + 1}/{num_epochs}], Step [{i + 1}/{len(validloader)}], Loss: {loss_value.item():.4f}')
            # Save Model with lowest validation Loss
            if loss_value.item() < best_valid_loss:
                print('Found new best validation loss')
                best_valid_loss = loss_value.item()
                is_best = True
                file_path = '../models/cnn_model.pth.tar'
                save_checkpoint({
                    'epoch': i_epoch + 1,
                    'state_dict': cnn_model.state_dict(),
                    'best_loss': best_valid_loss,
                    'optimizer': optimizer.state_dict(),
                }, is_best, filename=file_path)
            valid_loss.append(loss_value.item())
        
    print(f"Training Complete! Best Training Loss: {best_train_loss:.4f}, Best Validation Loss: {best_valid_loss:.4f}")


    # Run Classification on Test Set

