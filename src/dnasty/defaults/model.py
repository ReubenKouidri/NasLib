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


class ECGNet1d(nn.Sequential):
    """Small 1D CNN reference for raw multi-lead ECG (about 100k parameters).

    stem conv (stride 2) -> 3 x [conv -> BN -> ReLU -> max-pool] -> global
    average pooling -> linear. Same padding throughout, so any input length
    works; the output is raw logits for either loss.
    """

    def __init__(
        self, in_channels: int = 12, num_classes: int = 5, width: int = 32
    ) -> None:
        def block(cin: int, cout: int, kernel_size: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv1d(cin, cout, kernel_size, padding=kernel_size // 2, bias=False),
                nn.BatchNorm1d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
            )

        super().__init__(
            OrderedDict(
                [
                    (
                        "stem",
                        nn.Sequential(
                            nn.Conv1d(
                                in_channels, width, 7, stride=2, padding=3, bias=False
                            ),
                            nn.BatchNorm1d(width),
                            nn.ReLU(inplace=True),
                        ),
                    ),
                    ("block1", block(width, 2 * width, 5)),
                    ("block2", block(2 * width, 4 * width, 5)),
                    ("block3", block(4 * width, 4 * width, 3)),
                    ("gap", nn.AdaptiveAvgPool1d(1)),
                    ("flatten", nn.Flatten()),
                    ("head", nn.Linear(4 * width, num_classes)),
                ]
            )
        )
