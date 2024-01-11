from __future__ import annotations
import collections.abc as collections
from typing import Union

import abc
from abc import abstractmethod
import inspect
import random
import warnings
import copy
import torch.nn as nn
import dnasty.components as components

__all__ = [
    "LinearBlockGene",
    "ConvBlock2dGene",
    "MaxPool2dGene",
    "SpatialAttentionGene",
    "ChannelAttentionGene",
    "CBAMGene"
]

from torch.nn import Module

_allowed_activations = nn.modules.activation.__all__


class GeneBase(abc.ABC):
    """Base class for all Genes to inherit from"""

    def __init__(self, exons):
        if not isinstance(exons, collections.Mapping):
            raise TypeError(
                f"Exons must be a Mapping type, not {type(exons).__name__}.")
        self.exons = dict(exons)
        self.is_expressed = True

    @abstractmethod
    def mutate(self, *args, **kwargs) -> None:
        raise NotImplementedError("Mutate function must be implemented!")

    def __getattr__(self, name):
        cls = self.__class__
        if hasattr(cls, name):  # search static attributes
            return getattr(cls, name)
        elif name in self.exons:
            return self.exons[name]
        else:
            raise AttributeError(f"No attribute {name} in {cls.__name__}")

    def express(self) -> Module:
        return _build_layer(self)

    @staticmethod
    def _validate_feature(name: str, value: int, allowed_range: set) -> int:
        """
        Adjusts the feature to the nearest valid value if not in the allowed set

        :param name: Name of the feature.
        :param value: Value of the feature.
        :param allowed_range: Allowed set of values for the feature.
        :return: Adjusted feature value.
        """
        if value not in allowed_range:
            closest_val = min(allowed_range,
                              key=lambda x: abs(x - value))
            warnings.warn(f"{name} ({value}) not allowed, "
                          f"adjusting to nearest value: {closest_val}")
            return closest_val
        return value

    def __deepcopy__(self, memo=None):
        if memo is None:
            memo = {}
        cls = self.__class__
        exons_copy = copy.deepcopy(self.exons, memo)
        new_obj = cls(**exons_copy)
        memo[id(self)] = new_obj
        return new_obj

    def __len__(self) -> int:
        return len(self.exons)


class LinearBlockGene(GeneBase):
    allowed_features = set(range(9, 10_000))

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 activation: str | None = None,
                 dropout: bool = True) -> None:
        if activation is not None and activation not in _allowed_activations:
            raise ValueError(
                f"Activation {activation} not in {_allowed_activations}.")
        if not isinstance(dropout, bool):
            raise TypeError(
                f"dropout must be a boolean, not {type(dropout).__name__}.")
        in_features = self._validate_feature("in_features", in_features,
                                             LinearBlockGene.allowed_features)
        out_features = self._validate_feature("out_features", out_features,
                                              LinearBlockGene.allowed_features)
        exons = {"in_features": in_features,
                 "out_features": out_features,
                 "dropout": dropout,
                 "activation": activation}

        super().__init__(exons)

    def mutate(self) -> None:
        self.exons["dropout"] = not self.exons["dropout"]
        self.exons["out_features"] += random.randrange(-100, 100, 10)
        self.exons["out_features"] = (
            self._validate_feature("out_features", self.exons["out_features"],
                                   LinearBlockGene.allowed_features))

    @staticmethod
    def _validate_feature(name: str, value: int, allowed_range: set) -> int:
        return GeneBase._validate_feature(name, value, allowed_range)


class ConvBlock2dGene(GeneBase):
    allowed_channels = set(2 ** i for i in range(1, 8))
    allowed_kernel_size = set(range(2, 16))

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int | tuple,
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

        exons = {"in_channels": in_channels,
                 "out_channels": out_channels,
                 "kernel_size": kernel_size,
                 "activation": activation,
                 "batch_norm": batch_norm}

        super().__init__(exons)

    def mutate(self, fnc):
        # TODO: Implement
        pass

    @staticmethod
    def _validate_feature(name: str, value: int, allowed_range: set) -> int:
        return GeneBase._validate_feature(name, value, allowed_range)


class MaxPool2dGene(GeneBase):
    allowed_values = set(range(2, 10))

    def __init__(self,
                 kernel_size: Union[tuple, list, int],
                 stride: Union[tuple, list, int, None] = None) -> None:
        kernel_size = self._validate_feature("kernel_size", kernel_size,
                                             MaxPool2dGene.allowed_values)
        if stride is None:
            stride = kernel_size

        exons = {"kernel_size": kernel_size, "stride": stride}
        super().__init__(exons)

    def mutate(self, fnc):
        # TODO: Implement
        pass

    @staticmethod
    def _validate_feature(name: str,
                          value: int | tuple | list,
                          allowed_range: set) -> int | list:
        if isinstance(value, tuple) or isinstance(value, list):
            if len(value) > 2:
                raise ValueError("MaxPool2dGene only supports 2D pooling")
            return list(
                [GeneBase._validate_feature(name, v, allowed_range) for v in
                 value])

        return GeneBase._validate_feature(name, value, allowed_range)


class SpatialAttentionGene(GeneBase):
    allowed_values = set(range(2, 10))

    def __init__(self, kernel_size: int | tuple):
        self.kernel_size = self._validate_feature(
            "kernel_size", kernel_size,
            SpatialAttentionGene.allowed_values)

        super().__init__({"kernel_size": self.kernel_size})

    def mutate(self, *args, **kwargs) -> None:
        dk = 1 if random.random() < 0.5 else -1
        self.exons["kernel_size"] = self._validate_feature(
            "kernel_size",
            self.exons["kernel_size"] + dk,
            SpatialAttentionGene.allowed_values)

    @staticmethod
    def _validate_feature(name: str, value: int, allowed_range: set) -> int:
        return GeneBase._validate_feature(name, value, allowed_range)


class ChannelAttentionGene(GeneBase):
    allowed_range = set(range(2, 10))

    def __init__(self, se_ratio: int, in_channels: int = 1):
        """
        Args:
            se_ratio: the squeeze-excitation ratio.
            in_channels: this is not an independent var as is set to
                         match the output of the previous layer.
                         Therefore, it is not added to the exons
        """
        self.in_channels = in_channels
        self.se_ratio = self._validate_feature(
            "se_ratio", se_ratio,
            ChannelAttentionGene.allowed_range)
        super().__init__({"se_ratio": self.se_ratio})

    @staticmethod
    def _validate_feature(name: str, value: int, allowed_range: set) -> int:
        return super().validate(name, value, allowed_range)

    def mutate(self, *args, **kwargs) -> None:
        dk = 1 if random.random() < 0.5 else -1
        self.exons["kernel_size"] = self._validate_feature(
            "kernel_size",
            self.exons["kernel_size"] + dk,
            ChannelAttentionGene.allowed_values)


class CBAMGene(GeneBase):
    def __init__(self,
                 ca_gene: ChannelAttentionGene,
                 sa_gene: SpatialAttentionGene) -> None:
        super().__init__({"ca_gene": ca_gene, "sa_gene": sa_gene})

    def mutate(self, *args, **kwargs) -> None:
        self.ca_gene.mutate(*args, **kwargs)
        self.sa_gene.mutate(*args, **kwargs)


def _build_layer(gene: GeneBase) -> nn.Module:
    name = type(gene).__name__.replace("Gene", "")
    if hasattr(components, name):
        module = components
    elif hasattr(nn, name):
        module = nn
    else:
        raise AttributeError(
            f"No class {name} found in either torch.nn or the current module")

    sig = inspect.signature(getattr(module, name))
    params = [p for p in gene.exons.keys() if p in sig.parameters]
    kwargs = {p: gene.exons[p] for p in params}
    return getattr(module, name)(**kwargs)
