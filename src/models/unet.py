import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv

class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, init_features: int = 64, depth: int = 4):
        super().__init__()
        self.depth = depth

        self.inc = DoubleConv(in_channels, init_features)
        
        self.downs = nn.ModuleList()
        features = init_features
        for _ in range(depth):
            self.downs.append(Down(features, features*2))
            features *= 2

        self.ups = nn.ModuelsList()
        for _ in range(depth):
            self.ups.append(Up(features, features // 2))
            features //= 2

        self.outc = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inc(x)
        skips: list[torch.Tensor] = [x]

        for i in range(self.depth - 1):
            x = self.downs[i](x)
            skips.append(x)
        x = self.downs[-1](x)
        
        for i in range(self.depth):
            skip = skip[-(i+1)]
            x = self.ups[i](x, skip)

        return self.outc(x)
    