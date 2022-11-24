import torch
from torch import nn


class Flatten(nn.Module):
    @staticmethod
    def forward(x):
        return x.view(x.size(0), -1)


class ConvBlock2D(nn.Module):
    def __init__(
            self, in_planes, out_planes, kernel_size,
            stride=1, padding=0, relu=True, bn=True, bias=True
    ):
        super(ConvBlock2D, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.relu = nn.ReLU() if relu else None
        self.bn = nn.BatchNorm2d(self.out_planes, momentum=0.1, affine=True) if bn else None
        self.conv = nn.Conv2d(self.in_planes, self.out_planes, kernel_size=self.kernel_size,
                              stride=(self.stride, self.stride), bias=self.bias)

    def forward(self, x):
        x = self.conv(x)
        if self.relu is not None:
            x = self.relu(x)
        if self.bn is not None:
            x = self.bn(x)
        return x


class MaxPool2D(nn.Module):
    def __init__(self, kernel_size):
        super(MaxPool2D, self).__init__()
        self.kernel_size = kernel_size
        self.pool = nn.MaxPool2d(self.kernel_size)

    def forward(self, x):
        return self.pool(x)


class ChannelAttention(nn.Module):
    def __init__(
            self, in_channels, reduction_ratio
    ):
        super(ChannelAttention, self).__init__()
        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio
        self.d1 = nn.Linear(self.in_channels, self.in_channels // self.reduction_ratio)
        self.d2 = nn.Linear(self.in_channels // self.reduction_ratio, self.in_channels)
        self.relu = nn.ReLU()
        self.activation = nn.Sigmoid()

    def forward(self, input_tensor):
        gmp = self._global_max_pool(input_tensor)
        gmp = self.relu(self.d1(gmp))
        gmp = self.relu(self.d2(gmp))

        gap = self._global_av_pool(input_tensor)
        gap = self.relu(self.d1(gap))
        gap = self.relu(self.d2(gap))

        s = torch.add(gmp, gap)  # element-wise sum
        s = self.activation(s).unsqueeze(dim=2).unsqueeze(dim=3).expand_as(input_tensor)
        output = torch.mul(input_tensor, s)  # matrix dot product
        return output

    @staticmethod
    def _global_max_pool(data):
        """
            Take in a batch tensor (batch_size, channels, height, width)
            return a tensor of shape (batch_size, channels)
        """
        max_pooled = nn.AdaptiveMaxPool2d(output_size=1)(data)
        max_pooled = max_pooled.squeeze(dim=3).squeeze(dim=2)
        return max_pooled

    @staticmethod
    def _global_av_pool(data):
        """
            see _global_max_pool
        """
        av_pooled = nn.AdaptiveAvgPool2d(output_size=1)(data)
        av_pooled = av_pooled.squeeze(dim=3).squeeze(dim=2)
        return av_pooled


class ChannelPool(nn.Module):
    @staticmethod
    def forward(x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        # concatenates tensors to gain description of both max and mean features


class SpatialAttention(nn.Module):
    def __init__(
            self,
            kernel_size=4
    ):
        super(SpatialAttention, self).__init__()
        self.kernel_size = kernel_size  # they use a fixed value of 7 in the paper for 224x224 images
        self.compress = ChannelPool()
        self.spatial = ConvBlock2D(2, 1, self.kernel_size, stride=1, padding=(self.kernel_size-1) // 2, relu=False)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        a = self.activation(x_out)  # broadcasting
        return x * a


class CBAM(nn.Module):
    def __init__(
            self, in_channels, reduction_ratio=4, kernel_size=4, spatial=True
    ):
        super(CBAM, self).__init__()
        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio
        self.kernel_size = kernel_size
        self.spatial = spatial
        self.ChannelGate = ChannelAttention(in_channels, reduction_ratio)
        self.SpatialGate = SpatialAttention(kernel_size=kernel_size) if self.spatial else None

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if self.SpatialGate:
            x_out = self.SpatialGate(x_out)
        return x_out


class ResBlock1(nn.Module):
    def __init__(
            self, in_planes_0, out_planes_0, conv_kernel_size_0, att_kernel_size, mp_ker,
            reduction_ratio, cbam=True, spatial=False
    ):
        super(ResBlock1, self).__init__()
        self.in_planes_0 = in_planes_0
        self.out_planes_0 = out_planes_0
        self.conv_kernel_size_0 = conv_kernel_size_0
        self.att_kernel_size = att_kernel_size
        self.mp_ker = mp_ker
        self.reduction_ratio = reduction_ratio
        self.spatial = spatial
        self.mp = MaxPool2D(self.mp_ker)
        self.conv_0 = ConvBlock2D(
            in_planes=self.in_planes_0, out_planes=self.out_planes_0,
            kernel_size=self.conv_kernel_size_0
        )
        self.cbam = CBAM(
            in_channels=self.out_planes_0, reduction_ratio=self.reduction_ratio,
            kernel_size=self.att_kernel_size, spatial=self.spatial
        ) if cbam else None

    def forward(self, x):
        x = self.conv_0(x)
        x = self.mp(x)
        if self.cbam is not None:
            x = self.cbam(x)
        return x


class ResBlock2(nn.Module):
    def __init__(
            self, in_planes_0, out_planes_0, conv_kernel_size_0, att_kernel_size, mp_ker,
            in_planes_1, out_planes_1, conv_kernel_size_1, reduction_ratio,
            spatial=False, cbam=True
    ):
        super(ResBlock2, self).__init__()
        self.in_planes_0 = in_planes_0
        self.out_planes_0 = out_planes_0
        self.conv_kernel_size_0 = conv_kernel_size_0
        self.in_planes_1 = in_planes_1
        self.out_planes_1 = out_planes_1
        self.conv_kernel_size_1 = conv_kernel_size_1
        self.att_kernel_size = att_kernel_size
        self.mp_ker = mp_ker
        self.reduction_ratio = reduction_ratio
        self.spatial = spatial
        self.mp = MaxPool2D(self.mp_ker)
        self.conv_0 = ConvBlock2D(in_planes=self.in_planes_0, out_planes=self.out_planes_0,
                                  kernel_size=self.conv_kernel_size_0
                                  )
        self.conv_1 = ConvBlock2D(in_planes=self.out_planes_0, out_planes=self.out_planes_1,
                                  kernel_size=self.conv_kernel_size_1
                                  )
        self.cbam = CBAM(in_channels=self.out_planes_1, reduction_ratio=self.reduction_ratio,
                         kernel_size=self.att_kernel_size, spatial=self.spatial
                         ) if cbam else None

    def forward(self, x):
        x = self.conv_0(x)
        x = self.conv_1(x)
        x = self.mp(x)
        if self.cbam is not None:
            x = self.cbam(x)
        return x


class ResBlock3(nn.Module):
    def __init__(
            self,
            in_planes_0, out_planes_0, conv_kernel_size_0,
            in_planes_1, out_planes_1, conv_kernel_size_1,
            in_planes_2, out_planes_2, conv_kernel_size_2,
            mp_ker, att_kernel_size, reduction_ratio,
            spatial=False, cbam=True
    ):
        super(ResBlock3, self).__init__()
        self.in_planes_0 = in_planes_0
        self.out_planes_0 = out_planes_0
        self.conv_kernel_size_0 = conv_kernel_size_0
        self.in_planes_1 = in_planes_1
        self.out_planes_1 = out_planes_1
        self.conv_kernel_size_1 = conv_kernel_size_1
        self.in_planes_2 = in_planes_2
        self.out_planes_2 = out_planes_2
        self.conv_kernel_size_2 = conv_kernel_size_2
        self.att_kernel_size = att_kernel_size
        self.mp_ker = mp_ker
        self.reduction_ratio = reduction_ratio
        self.spatial = spatial
        self.mp = MaxPool2D(self.mp_ker)
        self.conv_0 = ConvBlock2D(
            in_planes=self.in_planes_0, out_planes=self.out_planes_0,
            kernel_size=self.conv_kernel_size_0
        )
        self.conv_1 = ConvBlock2D(
            in_planes=self.out_planes_0, out_planes=self.out_planes_1,
            kernel_size=self.conv_kernel_size_1
        )
        self.conv_2 = ConvBlock2D(
            in_planes=self.out_planes_1, out_planes=self.out_planes_2,
            kernel_size=self.conv_kernel_size_2
        )
        self.cbam = CBAM(
            in_channels=self.out_planes_2, reduction_ratio=self.reduction_ratio,
            kernel_size=self.att_kernel_size, spatial=self.spatial
        ) if cbam else None

    def forward(self, x):
        x = self.conv_0(x)
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.mp(x)
        if self.cbam is not None:
            x = self.cbam(x)
        return x


class DenseBlock(nn.Module):
    def __init__(
            self, in_features, out_features, relu=True, dropout=True
    ):
        super(DenseBlock, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.relu = nn.ReLU() if relu else None
        self.dropout = nn.Dropout(p=0.5) if dropout else None
        self.layer = nn.Linear(in_features=self.in_features, out_features=self.out_features)

    def forward(self, x):
        x = self.layer(x)
        if self.relu is not None:
            x = self.relu(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x
