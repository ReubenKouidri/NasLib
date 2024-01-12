from dnasty.components import ConvBlock2d, CBAM, LinearBlock, Flatten
import torch.nn as nn
from collections import OrderedDict


class S_2RB2D2(nn.Sequential):
    """sequential, 2x RB(2 conv each before attention) 2x Dense"""

    def __init__(self):
        architecture = OrderedDict([
            ("ConvBlock1", ConvBlock2d(1, 32, 10, bn=True, activation="ReLU")),
            ("ConvBlock2", ConvBlock2d(32, 32, 10, bn=True, activation="ReLU")),
            ("MP1", nn.MaxPool2d(2, 2)),
            ("CBAM1", CBAM(32, se_ratio=4, kernel_size=4)),
            ("ConvBlock3", ConvBlock2d(32, 32, 8, bn=True, activation="ReLU")),
            ("ConvBlock4", ConvBlock2d(32, 32, 4, bn=True, activation="ReLU")),
            ("MP2", nn.MaxPool2d(2, 2)),
            ("CBAM2", CBAM(32, se_ratio=4, kernel_size=4)),
            ("Flatten", Flatten()),
            ("DenseBlock1", LinearBlock(15488, 100, activation="ReLU")),
            ("DenseBlock2",
             LinearBlock(100, 9, activation="ReLU", dropout=False)),
            ("Softmax", nn.Softmax(dim=1))
        ])
        super(S_2RB2D2, self).__init__(architecture)
