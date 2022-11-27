from typing import Tuple, Optional, Union
import torch
from torch import Tensor
from torch import nn
from dnasty.my_types import k_size_t, stride_t, pad_t, dil_t, act_t


__allowed_activations__ = nn.modules.activation.__all__


def make_activation(name: str) -> act_t:
    if name in __allowed_activations__:
        return getattr(nn, name)()
    else:
        raise TypeError("Activation not valid!")


class MaxPool2D(nn.Module):
    def __init__(
            self,
            kernel_size: k_size_t,
            stride: Optional[stride_t] = 1,
            padding: Optional[Union[pad_t, str]] = 0,
            dilation: Optional[dil_t] = 1
    ) -> None:
        super(MaxPool2D, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.pool = nn.MaxPool2d(kernel_size=self.kernel_size, stride=self.stride,
                                 padding=self.padding, dilation=self.dilation)

    def forward(self, x: Tensor) -> Tensor:
        return self.pool(x)


class Flatten(nn.Module):
    """
    The -1 infers the size from the other dimension
    dim=0 is the batch dimension
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
    @staticmethod
    def forward(x: Tensor) -> Tensor:
        # (N, C, H, W) => channel dim = 1
        # torch.max returns type torch.return_types.max: first tensor contains the max values
        # whereas the second tensor is an index tensor showing which input channel the max value occurred in
        max_channel_pool = torch.max(x, dim=1)[0]
        max_channel_pool = max_channel_pool.unsqueeze(dim=1)
        av_channel_pool = torch.mean(x, dim=1)
        av_channel_pool = av_channel_pool.unsqueeze(dim=1)
        return torch.cat((av_channel_pool, max_channel_pool), dim=1)

        #return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        # concatenates tensors to gain description of both max and mean features


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: k_size_t) -> None:
        super(SpatialAttention, self).__init__()
        self.kernel_size = kernel_size
        self.compress = ChannelPool()
        self.conv = ConvBlock2D(in_channels=2, out_channels=1, kernel_size=self.kernel_size,
                                stride=1, padding='same', bn=False, activation="Sigmoid")

    def forward(self, x: Tensor) -> Tensor:
        sa = self.compress(x)
        sa = self.conv(sa)
        return torch.mul(x, sa)  # element-wise


class ConvBlock2D(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple],
            stride: Optional[Union[int, Tuple]] = 1,
            padding: Optional[Union[int, str]] = 0,
            dilation: Optional[int] = 1,
            groups: Optional[int] = 1,
            bias: Optional[bool] = True,
            bn: Optional[bool] = True,
            activation: str = None
    ):
        super(ConvBlock2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                              kernel_size=self.kernel_size, stride=self.stride,
                              padding=self.padding, dilation=self.dilation,
                              groups=self.groups, bias=self.bias)
        self.activation = make_activation(activation) if activation else None
        self.bn = nn.BatchNorm2d(self.out_channels, momentum=0.1, affine=True) if bn else False

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.activation(x) if self.activation is not None else x
        x = self.bn(x) if self.bn else x
        return x


class ChannelAttention(nn.Module):
    def __init__(
            self,
            in_channels: int,
            se_ratio: int,
            gmp_activation: Optional[str] = "ReLU",
            gap_activation: Optional[str] = "ReLU",
            out_activation: Optional[str] = "Sigmoid"
    ) -> None:
        super(ChannelAttention, self).__init__()
        self.in_channels = in_channels
        self.se_ratio = se_ratio
        self.gmp_activation = make_activation(gmp_activation)
        self.gap_activation = make_activation(gap_activation)
        self.out_activation = make_activation(out_activation)
        self.d1 = nn.Linear(in_features=self.in_channels, out_features=self.in_channels // self.se_ratio)
        self.d2 = nn.Linear(in_features=self.in_channels // self.se_ratio, out_features=self.in_channels)

    def forward(self, x: Tensor) -> Tensor:
        # global max pool the input feature map
        gmp = self._gmp(x)
        gmp = self.gmp_activation(self.d1(gmp))
        gmp = self.gmp_activation(self.d2(gmp))
        # global average pool the same input feature map
        gap = self._gap(x)
        gap = self.gap_activation(self.d1(gap))
        gap = self.gap_activation(self.d2(gap))
        # element-wise addition
        s = torch.add(gmp, gap)
        # apply activation and broadcast to input feature map size
        s = self.out_activation(s).unsqueeze(dim=2).unsqueeze(dim=3).expand_as(x)
        # matrix dot prod output map with input map
        return torch.mul(x, s)

    @staticmethod
    def _gmp(x: Tensor) -> Tensor:
        """
        TODO: correct docstring and add more description
        global max pool
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


class CBAM(nn.Module):
    def __init__(
            self,
            in_channels: int,
            se_ratio: int,
            kernel_size: Optional[k_size_t] = 4,
            spatial: Optional[bool] = True,
            channel: Optional[bool] = True
    ) -> None:
        super(CBAM, self).__init__()
        self.in_channels = in_channels
        self.se_ratio = se_ratio
        self.kernel_size = kernel_size
        self.spatial = spatial
        self.channel_gate = ChannelAttention(in_channels=self.in_channels, se_ratio=self.se_ratio) if channel else None
        self.spatial_gate = SpatialAttention(self.kernel_size) if spatial else None

    def forward(self, x: Tensor) -> Tensor:
        cb = x
        if self.channel_gate is not None:
            cb = self.channel_gate(cb)
        if self.spatial_gate is not None:
            cb = self.spatial_gate(cb)

        return torch.add(x, cb)


class DenseBlock(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            activation: Optional[str],
            dropout: Optional[bool] = True
    ) -> None:
        super(DenseBlock, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = make_activation(activation) if activation is not None else None
        self.dropout = nn.Dropout(p=0.5) if dropout else None
        self.layer = nn.Linear(in_features=self.in_features, out_features=self.out_features)

    def forward(self, x: Tensor) -> Tensor:
        x = self.layer(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x
