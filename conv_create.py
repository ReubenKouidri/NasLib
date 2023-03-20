from dnasty.genetics import ConvGene


class ConvBlock2D(nn.Sequential):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple],
            stride: Union[int, Tuple] | None = 1,
            padding: Union[int, str] | None = 0,
            dilation: int | None = 1,
            groups: int | None = 1,
            bias: bool | None = True,
            bn: bool | None = True,
            activation: str | None = None
    ) -> None:
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
        self.add_module("conv", self.conv)
        if activation:
            self.add_module(f"{activation}", make_activation(activation))
        if bn:
            self.add_module("batch_norm", nn.BatchNorm2d(self.out_channels, momentum=0.1, affine=True))