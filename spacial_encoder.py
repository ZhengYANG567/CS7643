import torch
import torch.nn as nn
from Unet import DoubleConv

class Spacial_Encoder(nn.Module):
    def __init__(self, in_channels=5, base_features=64, num_layers=4, use_skip=True):
        super().__init__()
        self.use_skip = use_skip
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        features = base_features
        for _ in range(num_layers):
            self.downs.append(DoubleConv(in_channels, features))
            in_channels = features
            features *= 2

        self.bottleneck = DoubleConv(in_channels, features)
        self.output_channels = features

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            if self.use_skip:
                skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        if self.use_skip:
            return x, skip_connections[::-1]
        else:
            return x, []
