"""
Based on a single level VQ-VAE from Jukebox:
https://arxiv.org/abs/2005.00341
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.channels = channels
        self.dilation = dilation

        self.dilated_conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            dilation=dilation,
            padding=dilation,
        )
        self.regular_conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x: torch.Tensor):
        """Note that the residual pathway is not built into forward()."""
        to_add = self.dilated_conv(x)
        to_add = self.regular_conv(to_add)
        return to_add


class EncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_residual_blocks: int,
    ):
        super().__init__()
        self.n_residual_blocks = n_residual_blocks
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.downsampling_conv = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self.residual_blocks = nn.ModuleList([])
        for i in range(n_residual_blocks):
            self.residual_blocks.append(
                ResidualBlock(channels=self.out_channels, dilation=3 ** (1 + i))
            )

    def forward(self, x: torch.Tensor):
        x = self.downsampling_conv(x)
        for i in range(self.n_residual_blocks):
            x = x + self.residual_blocks[i](x)

        return x


class Encoder(nn.Module):
    def __init__(self, channels: int, n_blocks: int):
        super().__init__()
        self.channels = channels
        self.blocks = [
            EncoderBlock(
                in_channels=1 if i == 0 else channels,
                out_channels=channels,
                n_residual_blocks=4,
            )
            for i in range(n_blocks)
        ]

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_residual_blocks: int,
    ):
        super().__init__()
        self.n_residual_blocks = n_residual_blocks
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.residual_blocks = nn.ModuleList([])
        for i in range(n_residual_blocks):
            self.residual_blocks.append(
                ResidualBlock(
                    channels=self.in_channels,
                    dilation=3 ** (i + 1),
                )
            )

        self.upsampling_conv = nn.ConvTranspose1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )

    def forward(self, x: torch.Tensor):
        for i in range(self.n_residual_blocks):
            x = x + self.residual_blocks[i](x)
        x = self.upsampling_conv(x)

        return x


class Decoder(nn.Module):
    def __init__(self, channels: int, n_blocks: int):
        super().__init__()
        self.channels = channels
        self.blocks = [
            DecoderBlock(
                in_channels=channels,
                out_channels=1 if i == n_blocks - 1 else channels,
                n_residual_blocks=4,
            )
            for i in range(n_blocks)
        ]

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
