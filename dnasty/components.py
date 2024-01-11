from __future__ import annotations
from typing import Union
from collections import OrderedDict
from dnasty.my_utils.types import size_2_t, size_2_opt_t

import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn

__all__ = [
    "LinearBlock",
    "ConvBlock2d",
    "Flatten",
    "ChannelPool",
    "SpatialAttention",
    "ChannelAttention",
    "CBAM"
]


class LinearBlock(nn.Sequential):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 dropout: bool = False,
                 activation: str | None = None) -> None:
        cell = OrderedDict()
        cell["linear"] = nn.Linear(in_features, out_features)
        if dropout:
            cell["dropout"] = nn.Dropout(p=0.5)
        if activation is not None:
            activation = getattr(nn, activation)()
            cell[f"{type(activation).__name__}"] = activation

        super().__init__(cell)


class ConvBlock2d(nn.Sequential):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: size_2_t,
            stride: size_2_opt_t | None = 1,
            padding: Union[str, size_2_t] | None = 0,
            groups: int | None = 1,
            bias: bool | None = True,
            bn: bool | None = True,
            activation: str | None = None
    ) -> None:
        layers = OrderedDict()
        layers["conv"] = nn.Conv2d(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   groups=groups,
                                   bias=bias)
        if bn:
            layers["batch_norm"] = nn.BatchNorm2d(out_channels, momentum=0.1,
                                                  affine=True)
        if activation:
            activation = getattr(nn, activation)()
            layers[f"{type(activation).__name__}"] = activation

        super(ConvBlock2d, self).__init__(layers)


class Flatten(nn.Module):
    """
    The -1 infers the size from the other dimension
    dim=0 is the batch dimension,
    so we are flattening the (N, C, H, W) tensor torch (N, C*H*W)

    Example:
        x = torch.randn((2, 3, 2, 2))
        print(x)
        print(x.view(x.size(dim=0), -1))
    """

    @staticmethod
    def forward(x: Tensor) -> Tensor:
        return x.view(x.size(0), -1)


class ChannelPool(nn.Module):
    """
    A module that performs channel pooling on an input tensor.
    The input tensor should have shape `(N, C, H, W)`, where
    `N` := batch size,
    `C` := number of channels,
    `H` := height, and
    `W` := width.

    Channel pooling computes the maximum and average values over the channel
    dimension separately, then concatenates the results along the channel
    dimension to produce an output tensor of shape `(N, 2, H, W)`,
    i.e. a description of both max and mean features along channel dim.

    Args:
        torch.Tensor: A tensor of shape `(N, C, H, W)`.

    Returns:
        torch.Tensor: A tensor of shape `(N, 2, H, W)`.

    Example::

        # Create a ChannelPool module
        channel_pool = ChannelPool()

        # Apply channel pooling to an input tensor
        x = torch.randn(16, 64, 32, 32)
        y = channel_pool(x)
        y.shape >> torch.Size([16, 2, 32, 32])
    """

    @staticmethod
    def forward(x: Tensor) -> Tensor:
        return torch.cat((torch.max(x, dim=1)[0].unsqueeze(dim=1),
                          torch.mean(x, dim=1).unsqueeze(dim=1)), dim=1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: size_2_t) -> None:
        super(SpatialAttention, self).__init__()
        self.spatial = ConvBlock2d(in_channels=2,
                                   out_channels=1,
                                   kernel_size=kernel_size,
                                   padding='same')
        self.compress = ChannelPool()

    def forward(self, x):
        x_compressed = self.compress(x)
        x_out = self.spatial(x_compressed)  # shape (N, 1, H, W)
        x_out = F.sigmoid(x_out)
        return torch.mul(x, x_out)


class ChannelAttention(nn.Module):
    def __init__(self, in_channels: int, se_ratio: int) -> None:
        super().__init__()
        self.gmp = nn.AdaptiveMaxPool2d(output_size=1)
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // se_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // se_ratio, in_channels),
            nn.ReLU()
        )

    def forward(self, x: Tensor) -> Tensor:
        gmp = self.gmp(x).squeeze(dim=-1).squeeze(dim=-1)
        gap = self.gap(x).squeeze(dim=-1).squeeze(dim=-1)
        mp = self.fc(gmp)
        ap = self.fc(gap)
        s = nn.Sigmoid()(torch.add(mp, ap))
        return x * s.unsqueeze(dim=-1).unsqueeze(dim=-1)  # .expand_as(x)


class CBAM(nn.Module):
    def __init__(
            self,
            in_channels: int,
            se_ratio: int,
            kernel_size: size_2_opt_t = 4,
            spatial: bool | None = True,
            channel: bool | None = True
    ) -> None:
        super(CBAM, self).__init__()
        self.in_channels = in_channels
        self.se_ratio = se_ratio
        self.kernel_size = kernel_size
        self.spatial = spatial
        self.channel_gate = ChannelAttention(in_channels=self.in_channels,
                                             se_ratio=self.se_ratio) if (
            channel) else None
        self.spatial_gate = SpatialAttention(
            self.kernel_size) if spatial else None

    def forward(self, x: Tensor) -> Tensor:
        cb = self.channel_gate(x) if self.channel_gate else x
        cb = self.spatial_gate(cb) if self.spatial_gate else cb
        return cb + x
