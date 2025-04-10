import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        """
        As implemented in the original paper (https://arxiv.org/pdf/1512.03385), a residual block
        consists of two consecutive convolution operations before adding the input to the output..

        The first convolution involves upsampling the number of feature channels,
        while the second convolution maintains the same dimensionality of the feature map.

        When adding the residual, it may be necessary to downsample in order to match the dimension of the 
        output resulting from the two convolution operations.
        """
        super(ResidualBlock, self).__init__()
        self.downsample = downsample

        self.conv1 = nn.Sequential(
         nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
         nn.BatchNorm2d(out_channels),
         nn.ReLU())

        self.conv2 = nn.Sequential(
         nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
         nn.BatchNorm2d(out_channels),
         nn.ReLU())

        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):

        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out = out + residual
        out = self.relu(out)
        
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10, batch_norm=True):
        """
        A ResNet consists of a Convolutional Network with a series of residual blocks.

        The first convolutional layer increases the number of feature maps to 64, and the subsequent layers
        consist of residual blocks. The number of feature maps is doubled after each layer, and the spatial
        dimensions are halved. The final layer is a fully connected layer that outputs the class scores.

        The block parameter specifies which ResidualBlock to use. The original paper introduces
        standard residual blocks and also "bottleneck" residual blocks.

        THe layers parameter specifies the number of residual blocks in each layer.
        For example, layers = [2, 2, 2, 2] corresponds to the original ResNet-18 architecture.
        """
        super(ResNet, self).__init__()
        self.inplanes = 64 # Variable to keep track of number input channels
        self.batch_norm = batch_norm

        self.conv1 = nn.Sequential(
                    nn.Conv2d(3, self.inplanes, kernel_size = 7, stride = 2, padding = 3),
                    nn.BatchNorm2d(self.inplanes),
                    nn.ReLU(),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Add Residual Blocks
        self.layer0 = self._add_residual(block, 64, layers[0], stride=1)
        self.layer1 = self._add_residual(block, 128, layers[1], stride=2)
        self.layer2 = self._add_residual(block, 256, layers[2], stride=2)
        self.layer3 = self._add_residual(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)

        self.fc = nn.Linear(512, num_classes)
    
    def _add_residual(self, block, out_channels, blocks, stride=1):
        """Add Residual Block
        
        Args:
            block: The type of residual block to add
            out_channels: The number of output channels for the residual block
            blocks: The number of blocks to add
            stride: The stride for the residual block
        """
        downsample = None
        if stride != 1 or self.inplanes != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )
        
        layers = []
        layers.append(block(self.inplanes, out_channels, stride, downsample))
        self.inplanes = out_channels
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, out_channels))

        return nn.Sequential(*layers)
    

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = self.fc(x.reshape(x.shape[0],-1))
        return x