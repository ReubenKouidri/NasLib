from __future__ import annotations
import collections.abc as collections

import abc
from abc import abstractmethod
import inspect
import random
import warnings
import copy
import torch.nn as nn
import dnasty.components as components
from dnasty.my_utils.types import size_2_opt_t, size_2_t

__all__ = [
    "LinearBlockGene",
    "ConvBlock2dGene",
    "MaxPool2dGene",
    "SpatialAttentionGene",
    "ChannelAttentionGene",
    "CBAMGene"
]

_allowed_activations = nn.modules.activation.__all__


class GeneBase(abc.ABC):
    """
    Base class for genetic representations of neural network components.

    `GeneBase` acts as an abstract base class for various types of 'genes',
    each representing a different neural network component or configuration.
    It provides a common interface and shared functionality for gene mutation,
    expression into PyTorch modules, and validation of gene features.

    Attributes:
        exons (dict): A dictionary representing the parts of the gene involved
                      in the construction of the corresponding nn.Module.
                      Only the exons are involved in crossover and mutation.

        is_active (bool): A flag controlling whether to express the gene -
                          useful in the Genome class.

    Args:
        exons (collections.Mapping): A mapping of gene feature names to their
        values.

    Raises:
        TypeError: If the provided exons are not in a mapping format.
    """
    __module_sig_cache = {}

    def __init__(self, exons) -> None:
        if not isinstance(exons, collections.Mapping):
            raise TypeError(
                f"Exons must be a Mapping type, not {type(exons).__name__}.")
        self.exons = dict(exons)
        self.is_active = True

    @abstractmethod
    def mutate(self, *args, **kwargs) -> None:
        raise NotImplementedError("Mutate function must be implemented!")

    def to_module(self) -> nn.Module:
        """
        Dynamically creates and returns a PyTorch module based on the
        gene's configuration.

        The module type is determined by the gene's class name and a
        matching class is searched for in `dnasty.components` and
        `torch.nn` modules.

        A class-level cache (`__module_sig_cache`) is used to
        store and reuse the signatures of module constructors,
        improving runtime performance by avoiding redundant introspection.

        Raises:
            AttributeError: If no corresponding class is found in either
            `dnasty.components` or `torch.nn` for the given gene type.

        :return:
            Module: A torch.nn.Module corresponding to the gene's type
            and configuration.

        Example::
            # Assuming a subclass of GeneBase 'ConvBlock2dGene' and
            corresponding 'ConvBlock2d' in dnasty.components
            >>> gene = ConvBlock2dGene({'in_channels': 32,
            ...                         'out_channels': 64,
            ...                         'kernel_size': 5,
            ...                         'activation': 'ReLU'})
            >>> my_module = gene.to_module()
            >>> print(my_module)
            ConvBlock2d(
                (conv): Conv2d(32, 64, kernel_size=(5, 5), stride=(1, 1))
                (ReLU): ReLU()
                (batch_norm): BatchNorm2d(64, eps=1e-05, momentum=0.1,
                                          affine=True,
                                          track_running_stats=True)
            )
        """
        name = type(self).__name__.replace("Gene", "")
        module = getattr(components, name, getattr(nn, name, None))
        if module is None:
            raise AttributeError(
                f"No class {name} found in either torch.nn or "
                f"dnasty.components")

        if name not in GeneBase.__module_sig_cache:
            GeneBase.__module_sig_cache[name] = inspect.signature(module)

        sig = GeneBase.__module_sig_cache[name]
        all_params = {**self.__dict__, **self.exons}
        params = {p: all_params[p] for p in all_params if p in sig.parameters}
        return module(**params)

    @staticmethod
    def _validate_feature(name: str, value: int, allowed_range: set) -> int:
        """
        Adjusts the feature to the nearest valid value if not in
        the allowed set.

        Args:
            name (str): The name of the feature.
            value (int): The value of the feature.
            allowed_range (set): The allowed set of values for the feature.

        Returns:
            Adjusted feature value if not in the allowed set else value.
        """
        if value not in allowed_range:
            closest_val = min(allowed_range,
                              key=lambda x: abs(x - value))
            warnings.warn(f"{name} ({value}) not allowed, "
                          f"adjusting to nearest value: {closest_val}")
            return closest_val
        return value

    def __getattr__(self, name):
        """
        The __getattr__ method is invoked by the interpreter when
        attribute lookup fails.

        Python checks if the instance has an attribute named x; if
        not, the search goes to the class (self.__class__), and then up the
        inheritance graph. If that fails, it looks for the attribute in the
        `exons` dictionary, and returns it accordingly.

        Args:
            name (str): The name of the attribute being accessed.

        Returns:
            The value of the attribute if it exists either as a class
            attribute or as a key in the `exons` dictionary.

        Raises:
            AttributeError: If the attribute is not found.
        """
        cls = type(self)
        if hasattr(cls, name):
            return getattr(cls, name)
        elif name in self.exons:
            return self.exons[name]
        else:
            raise AttributeError(
                f"No attribute {name} in {cls.__name__}")

    def __deepcopy__(self, memo=None):
        if memo is None:
            memo = {}
        cls = type(self)
        exons_copy = copy.deepcopy(self.exons, memo)
        new_obj = cls(**exons_copy)
        memo[id(self)] = new_obj
        return new_obj

    def __len__(self) -> int:
        return len(self.exons)


class LinearBlockGene(GeneBase):
    """
    LinearBlockGene encodes the standard configuration:
        nn.Linear -> Activation -> Dropout

    Attributes:
        allowed_features (set): A set defining the allowed values for
        the number of in- and out-features of the linear block.

    Args:
        in_features (int): The number of inputs to the linear layer.
        out_features (int): The output size of the linear layer.
        activation (str | None, optional): The activation function to be
         applied. Must be one of the allowed activations. Defaults to None.
        dropout (bool, optional): Flag to indicate whether dropout should be
        applied. Defaults to True.

    Raises:
        ValueError: If the provided activation function is not allowed.
        TypeError: If the dropout argument is not a boolean.
    """
    allowed_features = set(range(9, 10_000))

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 activation: str = None,
                 dropout: bool = True) -> None:

        if activation is not None and activation not in _allowed_activations:
            raise ValueError(
                f"Activation ({activation}) not in {_allowed_activations}.")
        if not isinstance(dropout, bool):
            raise TypeError(
                f"dropout must be a boolean, not {type(dropout).__name__}.")

        in_features = self._validate_feature("in_features",
                                             in_features,
                                             LinearBlockGene.allowed_features)
        out_features = self._validate_feature("out_features",
                                              out_features,
                                              LinearBlockGene.allowed_features)
        super().__init__({
            "in_features": in_features,
            "out_features": out_features,
            "dropout": dropout,
            "activation": activation
        })

    def mutate(self) -> None:
        self.dropout = not self.dropout
        self.out_features += random.randrange(-100, 100, 10)
        self.out_features = self._validate_feature(
            "out_features",
            self.out_features,
            LinearBlockGene.allowed_features)

    @staticmethod
    def _validate_feature(name: str, value: int, allowed_range: set) -> int:
        return GeneBase._validate_feature(name, value, allowed_range)


class ConvBlock2dGene(GeneBase):
    """
    A Gene encoding the common conv2d configuration:
        Conv2d -> Activation -> BatchNorm2d

    Attributes:
        allowed_channels (set): Defines the allowed range for the number
        of input and output channels.
        allowed_kernel_size (set): Defines the allowed range for the
        kernel size.

    Args:
        in_channels (int): Number of channels in the input image.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (size_2_t): Size of the convolving kernel.
        activation (str, optional): Name of the activation function to be
        applied. Must be one of the allowed activations. Defaults to "ReLU".
        batch_norm (bool | None, optional): Indicates whether batch
        normalization should be applied. Defaults to True.

    Raises:
        ValueError: If the provided activation is not in the list of
        allowed activations.
    """

    allowed_channels = set(2 ** i for i in range(1, 8))
    allowed_kernel_size = set(range(2, 16))

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: size_2_t,
                 activation: str = "ReLU",
                 batch_norm: bool | None = True) -> None:
        if activation not in _allowed_activations:
            raise ValueError(f"Unknown activation {activation}")

        in_channels = self._validate_feature("in_channels", in_channels,
                                             type(self).allowed_channels)
        out_channels = self._validate_feature("out_channels", out_channels,
                                              type(self).allowed_channels)
        kernel_size = self._validate_feature("kernel_size", kernel_size,
                                             type(self).allowed_kernel_size)

        super().__init__({"in_channels": in_channels,
                          "out_channels": out_channels,
                          "kernel_size": kernel_size,
                          "activation": activation,
                          "batch_norm": batch_norm})

    def mutate(self):
        # TODO: Implement
        pass

    @staticmethod
    def _validate_feature(name: str, value: int, allowed_range: set) -> int:
        return GeneBase._validate_feature(name, value, allowed_range)


class MaxPool2dGene(GeneBase):
    """
    Gene encoding a standard torch.nn.max_pool2d layer.

    Attributes:
        allowed_values (set): Defines the allowed range for the kernel size
        and stride values.

    Args:
        kernel_size (size_2_t): Size of the pooling window.
        stride (size_2_opt_t, optional): The stride of the window. Defaults
        to the same value as kernel_size.

    Raises:
        ValueError: If the provided kernel size is not within the allowed
        range, or dim > 2.

    Example:
        # Creating a MaxPool2dGene with a kernel size of 2x2 and default stride
        >>> gene = MaxPool2dGene(kernel_size=2)
        >>> module = gene.to_module()
        >>> isinstance(module, nn.MaxPool2d)
        True
        >>> print(module)
        MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1,
         ceil_mode=False)
    """
    allowed_values = set(range(2, 10))

    def __init__(self,
                 kernel_size: size_2_t,
                 stride: size_2_opt_t = None) -> None:
        kernel_size = self._validate_feature("kernel_size", kernel_size,
                                             MaxPool2dGene.allowed_values)
        if stride is None:
            stride = kernel_size
        else:
            stride = self._validate_feature("stride", stride,
                                            MaxPool2dGene.allowed_values)

        super().__init__({"kernel_size": kernel_size, "stride": stride})

    def mutate(self, fnc):
        # TODO: Implement
        pass

    @staticmethod
    def _validate_feature(name: str,
                          value: size_2_t,
                          allowed_range: set) -> int | list:
        if isinstance(value, tuple) or isinstance(value, list):
            if len(value) > 2:
                raise ValueError("MaxPool2dGene only supports 2D pooling")
            return list(
                [GeneBase._validate_feature(name, v, allowed_range) for v in
                 value])

        return GeneBase._validate_feature(name, value, allowed_range)


class SpatialAttentionGene(GeneBase):
    """
    Gene encoding a spatial attention module: https://arxiv.org/abs/1807.06521

    Attributes:
        allowed_values (set): Defines the allowed range for the kernel size
        of the spatial attention mechanism.

    Args:
        kernel_size (size_2_t): The size of the kernel to be used.
         Can be a single integer or a tuple of two integers.
    """
    allowed_values = set(range(2, 10))

    def __init__(self, kernel_size: size_2_t):
        kernel_size = self._validate_feature(
            "kernel_size", kernel_size,
            SpatialAttentionGene.allowed_values)

        super().__init__({"kernel_size": kernel_size})

    def mutate(self, *args, **kwargs) -> None:
        dk = 1 if random.random() < 0.5 else -1
        self.kernel_size = self._validate_feature(
            "kernel_size",
            self.kernel_size + dk,
            SpatialAttentionGene.allowed_values)

    @staticmethod
    def _validate_feature(name: str, value: int, allowed_range: set) -> int:
        return GeneBase._validate_feature(name, value, allowed_range)


class ChannelAttentionGene(GeneBase):
    """
    Gene encoding a channel attention module: https://arxiv.org/abs/1807.06521

    Attributes:
        allowed_values (set): Defines the allowed range for the squeeze

    Args:
        se_ratio: the squeeze-excitation ratio.
        in_channels: this is not an independent var as is set to match
        the output of the previous layer, and therefore, it is not
        added to the exons.
    """
    allowed_values = set(range(2, 16))

    def __init__(self, se_ratio: int, in_channels: int = 1):
        self.in_channels = in_channels
        se_ratio = self._validate_feature(
            "se_ratio", se_ratio,
            ChannelAttentionGene.allowed_values)
        super().__init__({"se_ratio": se_ratio})

    @staticmethod
    def _validate_feature(name: str, value: int, allowed_range: set) -> int:
        return GeneBase._validate_feature(name, value, allowed_range)

    def mutate(self, *args, **kwargs) -> None:
        dr = 1 if random.random() < 0.5 else -1
        self.se_ratio = self._validate_feature(
            "se_ratio",
            self.se_ratio + dr,
            ChannelAttentionGene.allowed_values)


class CBAMGene(GeneBase):
    """
    A Gene encoding the Convolutional Block Attention Module (CBAM):
    https://arxiv.org/abs/1807.06521.

    Args:
        channel_gene (ChannelAttentionGene): A gene to configure the
            channel attention mechanism.
        spatial_gene (SpatialAttentionGene): A gene to configure the
            spatial attention mechanism.

    Attributes:
        channel_gene (ChannelAttentionGene): Stored as an attribute for
        the moment so that these genes can be activated later.
        This might change.
        spatial_gene (SpatialAttentionGene): Same as above...
    """

    def __init__(self,
                 channel_gene: ChannelAttentionGene,
                 spatial_gene: SpatialAttentionGene):
        self.channel_gene = channel_gene
        self.spatial_gene = spatial_gene
        super().__init__({"in_channels": channel_gene.in_channels,
                          "se_ratio": channel_gene.se_ratio,
                          "kernel_size": spatial_gene.kernel_size})

    def mutate(self, *args, **kwargs) -> None:
        self.channel_gene.mutate(*args, **kwargs)
        self.spatial_gene.mutate(*args, **kwargs)
