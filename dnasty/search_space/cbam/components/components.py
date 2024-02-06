from __future__ import annotations
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from dnasty.my_utils import size_2_t, size_2_opt_t
from dnasty.search_space.common import Flatten, ConvBlock2d


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
        >>> channel_pool = ChannelPool()
        >>> x = torch.randn(16, 64, 32, 32)
        >>> y = channel_pool(x)
        >>> y.shape
        torch.Size([16, 2, 32, 32])
    """

    @staticmethod
    def forward(x: Tensor) -> Tensor:
        return torch.cat((torch.max(x, dim=1)[0].unsqueeze(dim=1),
                          torch.mean(x, dim=1).unsqueeze(dim=1)), dim=1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: size_2_t) -> None:
        super(SpatialAttention, self).__init__()
        self.conv = ConvBlock2d(in_channels=2,
                                out_channels=1,
                                kernel_size=kernel_size,
                                padding='same',
                                activation='Sigmoid')
        self.compress = ChannelPool()

    def forward(self, x):
        x_compressed = self.compress(x)
        x_out = self.conv(x_compressed).expand_as(x)
        return torch.mul(x, x_out)


class ChannelAttention(nn.Module):
    def __init__(self, in_channels: int, se_ratio: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(in_channels, max(in_channels // se_ratio, 1)),
            nn.ReLU(),
            nn.Linear(max(in_channels // se_ratio, 1), in_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        assert x.dim() == 4
        kernel_size = x.size()[2:]
        gmp = F.max_pool2d(x, kernel_size=kernel_size, stride=kernel_size)
        gap = F.avg_pool2d(x, kernel_size=kernel_size, stride=kernel_size)
        gmp = self.mlp(gmp)
        gap = self.mlp(gap)
        combined = torch.add(gap, gmp)
        combined = F.sigmoid(combined).unsqueeze(-1).unsqueeze(-1).expand_as(x)
        return torch.mul(x, combined)


class CBAM(nn.Module):
    def __init__(
            self,
            in_channels: int,
            se_ratio: int,
            kernel_size: size_2_opt_t = 4,
            spatial: bool = True,
            channel: bool = True
    ) -> None:
        super(CBAM, self).__init__()
        self.spatial = spatial
        self.channel = channel
        self.channel_gate = ChannelAttention(
            in_channels=in_channels,
            se_ratio=se_ratio) if self.channel else None
        self.spatial_gate = SpatialAttention(kernel_size) if spatial else None

    def forward(self, x: Tensor) -> Tensor:
        out = self.channel_gate(x) if self.channel_gate else x
        out = self.spatial_gate(out) if self.spatial_gate else out
        return torch.add(out, x)
