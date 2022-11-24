from typing import Tuple, Any, Optional, Union, TypeVar, TypeAlias
import torch
from torch import Tensor, Module
from torch import nn


class Flatten(nn.Module):
    """
    TODO:
        - add type hints
        - write documentation for how this works
    """
    @staticmethod
    def forward(x: Tensor):
        return x.view(x.size(0), -1)


class ConvLayer(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: k_size_t,
            stride: stride_t = 1,
            padding: Optional[int] = 1,
            dilation: Optional[int] = 1,
            groups: Optional[int] = 1,
            bias: Optional[bool] = True
    ):
        super(ConvLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                              kernel_size=kernel_size, stride=self.stride,
                              padding=self.padding, dilation=self.dilation,
                              groups=self.groups, bias=self.bias)

    def forward(self, x):
        return self.conv(x)


class MaxPool2D(nn.Module):
    def __init__(
            self,
            size: k_size_t,
            stride: Optional[stride_t],
            padding: Optional[pad_t],
            dilation: Optional[dil_t]
    ):
        super(MaxPool2D, self).__init__()
        self.kernel_size = size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.pool = nn.MaxPool2d(kernel_size=self.kernel_size, stride=self.stride,
                                 padding=self.padding, dilation=self.dilation)

    def forward(self, x: Tensor):
        return self.pool(x)


class ChannelAttention(nn.Module):
    def __init__(
            self,
            in_channels: int,
            se_ratio: int,
            gmp_activation: act_t,
            gap_activation: act_t,
            out_activation: act_t
    ) -> None:
        super(ChannelAttention, self).__init__()
        self.in_channels = in_channels
        self.se_ratio = se_ratio
        self.gmp_activation = gmp_activation
        self.gap_activation = gap_activation
        self.out_activation = out_activation
        self.d1 = nn.Linear(in_features=self.in_channels, out_features=self.in_channels // self.se_ratio)
        self.d2 = nn.Linear(in_features=self.in_channels // self.se_ratio, out_features=self.in_channels)

    def forward(self, x: Tensor) -> Tensor:
        gmp = self._gmp(x)
        gmp = self.gmp_activation(self.d1(gmp))
        gmp = self.gmp_activation(self.d2(gmp))

        gap = self._gap(x)
        gap = self.gap_activation(self.d1(gap))
        gap = self.gap_activation(self.d2(gap))

        s = torch.add(gmp, gap)
        s = self.out_activation(s).unsqueeze(dim=2).unsqueeze(dim=3).expand_as(x)  # resizing to same as input
        return torch.mul(x, s)  # matrix dot prod

    @staticmethod
    def _gmp(x: Tensor) -> Tensor:
        """
        TODO: correct docstring and add more description
        :param x: Tensor of form (batch_size, channels, height, width)
        :return: Tensor of form (batch_size, channels)
        """
        mp = nn.AdaptiveMaxPool2d(output_size=1)(x)
        mp = mp.squeeze(dim=3).squeeze(dim=2)
        return mp

    @staticmethod
    def _gap(x: Tensor) -> Tensor:
        """
        TODO: correct docstring and add more description
        global average pool on input Tensor
        :param x: 4D Tensor of feature maps
        :return: 2D reduced Tensor of average pooled feature maps
        """
        av = nn.AdaptiveAvgPool2d(output_size=1)(x)
        av = av.squeeze(dim=3).squeeze(dim=2)
        return av


class ChannelPool(nn.Module):
    ...


class SpatialAttention(nn.Module):
    ...


class CBAM(nn.Module):
    def __init__(
            self,
            in_channels: int,
            se_ratio: int,
            kernel_size: Optional[k_size_t] = 4,
            spatial: Optional[bool] = True
    ):
        super(CBAM, self).__init__()
        self.in_channels = in_channels
        self.se_ratio = se_ratio
        self.kernel_size = kernel_size
        self.spatial = spatial

    def build(self): ...
























