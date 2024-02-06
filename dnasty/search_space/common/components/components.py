from __future__ import annotations
from typing import Optional, Union
from torch import Tensor
from torch import nn
from dnasty.my_utils import size_2_t, size_2_opt_t, act_t
from dnasty.search_space.common.utils import get_activation


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
        activation = get_activation(activation)
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
        activation = get_activation(activation)
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
