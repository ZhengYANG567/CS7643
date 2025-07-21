import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import pickle
import os
from util import generate_checksum

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=24, out_channels=1, num_layers=4, features_start=64):
        super().__init__()
        features = features_start
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part
        for _ in range(num_layers):
            self.downs.append(DoubleConv(in_channels, features))
            in_channels = features
            features *= 2

        # Bottleneck
        self.bottleneck = DoubleConv(in_channels, features)

        # Up part
        for _ in range(num_layers):
            features //= 2
            self.ups.append(
                nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(features * 2, features))

        self.final_conv = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip_connection = skip_connections[i // 2]

            if x.shape != skip_connection.shape:
                x = torch.nn.functional.interpolate(x, size=skip_connection.shape[2:])

            x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[i + 1](x)

        return self.final_conv(x)

# Below: train_model, load_model, try_load_model, plot_losses, evaluate_model can be reused from CNN.py.
# You may copy them over and change CNN_FC() to UNet() accordingly.
