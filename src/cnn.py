import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self, num_filters_in=6,num_filters_out=16,
                 kernel=5, n_linear_first=120, n_linear_second=84,
                 batch_norm=False):
        super().__init__()
        self.batch_norm = batch_norm
        if batch_norm:
            self.bn1 = nn.BatchNorm2d(num_filters_in)
            self.bn2 = nn.BatchNorm2d(num_filters_out)

        self.conv1 = nn.Conv2d(3, num_filters_in, kernel)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(num_filters_in, num_filters_out, kernel)
        self.fc1 = nn.Linear(num_filters_out * kernel * kernel, 120)
        self.fc2 = nn.Linear(n_linear_first, n_linear_second)
        self.fc3 = nn.Linear(n_linear_second, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        if self.batch_norm:
            x = self.bn1(x)
        x = self.pool(F.relu(self.conv2(x)))
        if self.batch_norm:
            x = self.bn2(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x