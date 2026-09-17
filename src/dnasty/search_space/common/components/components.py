from __future__ import annotations

from torch import Tensor, nn

from dnasty.search_space.common.utils import get_activation
from dnasty.utils.types import act_t, size_2_t


class LinearBlock(nn.Sequential):
    """Linear -> [Activation] -> [Dropout(0.5)].

    Pass ``activation=None`` for the output layer so the network emits raw
    logits; ``nn.CrossEntropyLoss`` applies log-softmax itself.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: bool = False,
        activation: act_t = None,
    ) -> None:
        super().__init__()
        self.add_module("linear", nn.Linear(in_features, out_features))
        act = get_activation(activation)
        if act is not None:
            self.add_module(type(act).__name__, act)
        if dropout:
            self.add_module("dropout", nn.Dropout(p=0.5))


class ConvBlock2d(nn.Sequential):
    """Conv2d -> [BatchNorm2d] -> [Activation] (the ResNet/CBAM ordering)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: size_2_t,
        stride: size_2_t = 1,
        padding: str | size_2_t = 0,
        activation: act_t = "ReLU",
        batch_norm: bool = True,
    ) -> None:
        super().__init__()
        self.add_module(
            "conv",
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=not batch_norm,
            ),
        )
        if batch_norm:
            self.add_module(
                "batch_norm", nn.BatchNorm2d(out_channels, momentum=0.1, affine=True)
            )
        act = get_activation(activation)
        if act is not None:
            self.add_module(type(act).__name__, act)


class Flatten(nn.Module):
    """(N, C, H, W) -> (N, C*H*W)."""

    @staticmethod
    def forward(x: Tensor) -> Tensor:
        return x.view(x.size(0), -1)
