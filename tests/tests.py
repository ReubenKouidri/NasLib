import torch
import torch.nn as nn
from dnasty.components import ResBlock2, ConvBlock2D, DenseBlock, Flatten, Tensor, CBAM


"""class Model(nn.Module):
    RB2_1 = ResBlock2(in_channels_0=1, out_channels_0=32, out_channels_1=32,
                      conv_kernel_size_0=10, conv_kernel_size_1=10,
                      att_kernel_size=4, mp_ker_size=2, se_ratio=4,
                      conv_act_0='ReLU', conv_act_1='ReLU',
                      bn_0=True, bn_1=True, cbam=True, spatial=True)
    RB2_2 = ResBlock2(in_channels_0=32, out_channels_0=32, out_channels_1=32,
                      conv_kernel_size_0=8, conv_kernel_size_1=4,
                      att_kernel_size=4, mp_ker_size=2, se_ratio=4,
                      conv_act_0='ReLU', conv_act_1='ReLU',
                      bn_0=True, bn_1=True, cbam=True, spatial=True)
    d1 = DenseBlock(in_features=15488, out_features=100, activation='ReLU', dropout=True)
    d2 = DenseBlock(in_features=100, out_features=9, activation='ReLU', dropout=False)
    out_act = nn.Softmax(dim=1)

    def forward(self, x: Tensor) -> Tensor:
        o = self.RB2_1(x)
        o = self.RB2_2(o)
        o = Flatten()(o)
        o = self.d1(o)
        o = self.d2(o)
        return self.out_act(o)
"""

"""m = Model()
tensor = torch.randn(10, 1, 128, 128)
m(tensor)"""


class M2(nn.Module):
    def __init__(self):
        super(M2, self).__init__()
        self.attention = True
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.dense_1 = nn.Linear(in_features=15488, out_features=100)
        self.dense_2 = nn.Linear(in_features=100, out_features=9)
        self.dropout = nn.Dropout()
        self.softmax = nn.Softmax(dim=1)
        self.layer_1 = ConvBlock2D(in_channels=1, out_channels=32, kernel_size=10, bn=True, activation='ReLU')
        self.layer_2 = ConvBlock2D(in_channels=32, out_channels=32, kernel_size=10, bn=True, activation='ReLU')
        self.layer_3 = ConvBlock2D(in_channels=32, out_channels=32, kernel_size=8, bn=True, activation='ReLU')
        self.layer_4 = ConvBlock2D(in_channels=32, out_channels=32, kernel_size=4, bn=True, activation='ReLU')
        self.block_1 = nn.Sequential(self.layer_1, self.layer_2, self.maxpool1)
        self.block_2 = nn.Sequential(self.layer_3, self.layer_4, self.maxpool2)
        self.second_last = nn.Sequential(self.dense_1, self.relu, self.dropout)
        self.last = nn.Sequential(self.dense_2, self.relu, self.softmax)
        self.channel_att = CBAM(in_channels=32, se_ratio=4)
        self.channel_att2 = CBAM(in_channels=32, se_ratio=4)

    def forward(self, t):
        t = self.block_1(t)
        if self.attention:
            t = self.channel_att(t)

        t = self.block_2(t)
        if self.attention:
            t = self.channel_att(t)

        t = Flatten()(t)
        t = self.second_last(t)
        t = self.last(t)
        return t


m = M2()
tensor = torch.randn(10, 1, 128, 128)
print(tensor)
print(m(tensor))

