import torch
import torch.nn as nn
from dnasty.components import ConvBlock2D, DenseBlock, Flatten, Tensor, CBAM


class M2(nn.Module):
    def __init__(self):
        super(M2, self).__init__()
        self.cbam = True
        self.MP1 = nn.MaxPool2d(kernel_size=2)
        self.MP2 = nn.MaxPool2d(kernel_size=2)
        self.CB1 = ConvBlock2D(in_channels=1, out_channels=32, kernel_size=10, bn=True, activation='ReLU')
        self.CB2 = ConvBlock2D(in_channels=32, out_channels=32, kernel_size=10, bn=True, activation='ReLU')
        self.CB3 = ConvBlock2D(in_channels=32, out_channels=32, kernel_size=8, bn=True, activation='ReLU')
        self.CB4 = ConvBlock2D(in_channels=32, out_channels=32, kernel_size=4, bn=True, activation='ReLU')
        self.CBAM1 = CBAM(in_channels=32, se_ratio=4, kernel_size=4, spatial=True, channel=True)
        self.CBAM2 = CBAM(in_channels=32, se_ratio=4, kernel_size=4, spatial=True, channel=True)
        self.flatten = Flatten()
        self.DB1 = DenseBlock(in_features=15488, out_features=100, activation='ReLU', dropout=True)
        self.DB2 = DenseBlock(in_features=100, out_features=9, activation='ReLU', dropout=False)
        self.softmax = nn.Softmax(dim=1)

    def build_model(self) -> nn.Module:
        model = nn.Sequential()
        model.add_module("CB1", self.CB1)
        model.add_module("CB2", self.CB2)
        model.add_module("CBAM1", self.CBAM1) if self.cbam else ...
        model.add_module("MP1", self.MP1)
        model.add_module("CB3", self.CB3)
        model.add_module("CB4", self.CB4)
        model.add_module("CBAM2", self.CBAM2) if self.cbam else ...
        model.add_module("MP2", self.MP2)
        model.add_module("Flatten", self.flatten)
        model.add_module("DB1", self.DB1)
        model.add_module("DB2", self.DB2)
        model.add_module("softmax", self.softmax)
        return model

    def forward(self, t: Tensor) -> Tensor:
        return self.build_model()(t)


m = M2()
tensor = torch.randn(10, 1, 128, 128)
print(tensor)
print(m(tensor))
