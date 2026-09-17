from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from dnasty.search_space.common import ConvBlock2d, Flatten
from dnasty.utils.types import size_2_t


class ChannelPool(nn.Module):
    """Concatenate channel-wise max and mean: (N, C, H, W) -> (N, 2, H, W)."""

    @staticmethod
    def forward(x: Tensor) -> Tensor:
        return torch.cat(
            (torch.max(x, dim=1)[0].unsqueeze(1), torch.mean(x, dim=1).unsqueeze(1)),
            dim=1,
        )


class SpatialAttention(nn.Module):
    """CBAM spatial gate: ChannelPool -> Conv -> BN -> Sigmoid, applied as a mask."""

    def __init__(self, kernel_size: size_2_t) -> None:
        super().__init__()
        self.compress = ChannelPool()
        self.conv = ConvBlock2d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            padding="same",
            activation="Sigmoid",
            batch_norm=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        gate = self.conv(self.compress(x)).expand_as(x)
        return x * gate


class ChannelAttention(nn.Module):
    """CBAM channel gate: shared MLP over global max- and avg-pooled features."""

    def __init__(self, in_channels: int, se_ratio: int) -> None:
        super().__init__()
        hidden = max(in_channels // se_ratio, 1)
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(in_channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, in_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        assert x.dim() == 4
        kernel_size = x.size()[2:]
        gmp = self.mlp(F.max_pool2d(x, kernel_size=kernel_size, stride=kernel_size))
        gap = self.mlp(F.avg_pool2d(x, kernel_size=kernel_size, stride=kernel_size))
        gate = torch.sigmoid(gap + gmp).unsqueeze(-1).unsqueeze(-1).expand_as(x)
        return x * gate


class CBAM(nn.Module):
    """Channel then spatial attention with a residual connection around both."""

    def __init__(
        self,
        in_channels: int,
        se_ratio: int,
        kernel_size: size_2_t = 4,
        spatial: bool = True,
        channel: bool = True,
    ) -> None:
        super().__init__()
        self.channel_gate = (
            ChannelAttention(in_channels=in_channels, se_ratio=se_ratio)
            if channel
            else None
        )
        self.spatial_gate = SpatialAttention(kernel_size) if spatial else None

    def forward(self, x: Tensor) -> Tensor:
        out = self.channel_gate(x) if self.channel_gate is not None else x
        out = self.spatial_gate(out) if self.spatial_gate is not None else out
        return out + x
