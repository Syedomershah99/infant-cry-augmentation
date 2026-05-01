"""Small CNN classifier on log-mel spectrograms.

Designed for the donateacry scale (~457 clips after standard splits): a deep
ResNet would massively overfit, so this is a compact 4-block conv net with
batch norm and global average pooling. Parameters: ~0.5M.

Input:  (B, 1, 64, 128) standardized log-mel
Output: (B, 5) logits over LABELS = [belly_pain, burping, discomfort, hungry, tired]
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        x = self.pool(x)
        return self.drop(x)


class CryCNN(nn.Module):
    def __init__(self, num_classes: int = 5, base_channels: int = 32, dropout: float = 0.2):
        super().__init__()
        c = base_channels
        self.block1 = ConvBlock(1, c, dropout=dropout)
        self.block2 = ConvBlock(c, c * 2, dropout=dropout)
        self.block3 = ConvBlock(c * 2, c * 4, dropout=dropout)
        self.block4 = ConvBlock(c * 4, c * 8, dropout=dropout)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c * 8, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return self.head(x)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
