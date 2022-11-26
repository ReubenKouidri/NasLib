import torch
import torch.nn as nn
from dnasty.components import ResBlock2, DenseBlock, Flatten, Tensor


class Model(nn.Module):
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


m = Model()
tensor = torch.randn(10, 1, 128, 128)
m(tensor)


"""
self.attention = attention
self.relu = nn.ReLU()
self.maxpool1 = nn.MaxPool2d(kernel_size=2)
self.maxpool2 = nn.MaxPool2d(kernel_size=2)
self.dense_1 = nn.Linear(in_features=15488, out_features=100)
self.dense_2 = nn.Linear(in_features=100, out_features=9)
self.dropout = nn.Dropout()
self.softmax = nn.Softmax(dim=1)
self.layer_1 = self.basic_conv2d(in_planes=1, out_planes=32, kernel_size=10, bn=True, relu=True)
self.layer_2 = self.basic_conv2d(in_planes=32, out_planes=32, kernel_size=10, bn=True, relu=True)
self.layer_3 = self.basic_conv2d(in_planes=32, out_planes=32, kernel_size=8, bn=True, relu=True)
self.layer_4 = self.basic_conv2d(in_planes=32, out_planes=32, kernel_size=4, bn=True, relu=True)
self.block_1 = nn.Sequential(self.layer_1, self.layer_2, self.maxpool1)
self.block_2 = nn.Sequential(self.layer_3, self.layer_4, self.maxpool2)
self.second_last = nn.Sequential(self.dense_1, self.relu, self.dropout)
self.last = nn.Sequential(self.dense_2, self.relu, self.softmax)


def forward(self, t):
    t = self.block_1(t)
    if self.attention:
        t = self.channel_att(t, 32, 4)

    t = self.block_2(t)
    if self.attention:
        t = self.channel_att(t, 32, 4)

    t = self._flatten(t)
    t = self.second_last(t)
    t = self.last(t)
    return t
"""