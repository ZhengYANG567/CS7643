import torch
import torch.nn as nn
from Unet import DoubleConv

class Unet_Decoder(nn.Module):
    def __init__(self, out_channels=1, base_features=64, num_layers=4, bottleneck_channels=1024, use_skip=True):
        super().__init__()
        self.use_skip = use_skip
        self.ups = nn.ModuleList()

        features = bottleneck_channels
        for _ in range(num_layers):
            self.ups.append(nn.ConvTranspose2d(features, features // 2, kernel_size=2, stride=2))
            conv_in_channels = features if not self.use_skip else features
            conv_out_channels = features // 2
            self.ups.append(DoubleConv(conv_in_channels, conv_out_channels))
            features //= 2

        self.final_conv = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, x, skip_connections):
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)

            if self.use_skip and i // 2 < len(skip_connections):
                skip = skip_connections[i // 2]
                if x.shape != skip.shape:
                    x = torch.nn.functional.interpolate(x, size=skip.shape[2:])
                x = torch.cat((skip, x), dim=1)

            x = self.ups[i + 1](x)

        return self.final_conv(x)
