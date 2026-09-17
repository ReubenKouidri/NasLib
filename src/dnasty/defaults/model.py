from collections import OrderedDict

import torch.nn as nn

from dnasty.search_space.cbam import CBAM
from dnasty.search_space.common import ConvBlock2d, Flatten, LinearBlock


class S_2RB2D2(nn.Sequential):
    """Hand-designed reference: 2x (2 conv -> pool -> CBAM), 2x dense, logits out."""

    def __init__(self, num_classes: int = 9) -> None:
        architecture = OrderedDict(
            [
                ("ConvBlock1", ConvBlock2d(1, 32, 10, activation="ReLU")),
                ("ConvBlock2", ConvBlock2d(32, 32, 10, activation="ReLU")),
                ("MP1", nn.MaxPool2d(2, 2)),
                ("CBAM1", CBAM(32, se_ratio=4, kernel_size=4)),
                ("ConvBlock3", ConvBlock2d(32, 32, 8, activation="ReLU")),
                ("ConvBlock4", ConvBlock2d(32, 32, 4, activation="ReLU")),
                ("MP2", nn.MaxPool2d(2, 2)),
                ("CBAM2", CBAM(32, se_ratio=4, kernel_size=4)),
                ("Flatten", Flatten()),
                (
                    "DenseBlock1",
                    LinearBlock(15488, 100, dropout=True, activation="ReLU"),
                ),
                ("DenseBlock2", LinearBlock(100, num_classes, dropout=False)),
            ]
        )
        super().__init__(architecture)
