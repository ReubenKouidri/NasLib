from __future__ import annotations
from typing import Optional, Union
from dnasty.my_utils import size_2_t, size_2_opt_t, act_t

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F


def _get_activation(activation) -> nn.Module:
    if activation is None:
        return nn.ReLU()
    elif isinstance(activation, str):
        activation_module = getattr(nn, activation)()
    elif isinstance(activation, nn.Module):
        activation_module = activation
    else:
        raise TypeError("Activation must be a string, nn.Module, or None.")
    return activation_module


class LinearBlock(nn.Sequential):
    """
    Creates a sequential block: (Linear -> Activation -> Dropout)
    Dropout optional with default probability of 0.5
    (false for output layer).

    Args:
        in_features: The number of inputs to the linear layer.
        out_features: The number of outputs from the linear layer.
        dropout: If True, a dropout layer is added.
        activation: The activation function (as a string or nn.Module).

    Raises:
        TypeError: If the activation type is not supported.
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 dropout: bool = False,
                 activation: Optional[str, nn.Module] = None) -> None:
        super().__init__()
        self.add_module("linear", nn.Linear(in_features, out_features))
        activation = _get_activation(activation)
        self.add_module(type(activation).__name__, activation)
        if dropout:
            self.add_module("dropout", nn.Dropout(p=0.5))


class ConvBlock2d(nn.Sequential):
    """
    Constructs a convolutional block with batch normalisation and activation.

    Args:
        in_channels: Number of channels in the input.
        out_channels: Number of channels produced by the convolution.
        kernel_size: Size of the convolving kernel.
        stride: Stride of the convolution.
        padding: Padding added to all four sides of the input.
        activation: The activation function (as a string or nn.Module).

    Raises:
        ValueError: If padding type is not supported.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: size_2_t,
            stride: size_2_opt_t = 1,
            padding: Union[str, size_2_t] = 0,
            activation: act_t = None) -> None:
        super().__init__()
        self.add_module("conv",
                        nn.Conv2d(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=kernel_size,
                                  stride=stride,
                                  padding=padding))
        activation = _get_activation(activation)
        self.add_module(f"{type(activation).__name__}", activation)
        self.add_module("batch_norm",
                        nn.BatchNorm2d(out_channels, momentum=0.1, affine=True))


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
