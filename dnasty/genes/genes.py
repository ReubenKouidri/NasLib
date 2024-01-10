from __future__ import annotations
import collections.abc as collections

import abc
from abc import abstractmethod
import inspect
import random
import warnings
import copy
import torch.nn as nn

__all__ = [
    "LinearBlockGene",
    "ConvBlock2dGene",
    "MaxPool2dGene",
    "FlattenGene",
    "SpatialAttentionGene",
    "CBAMGene"
]

_allowed_activations = nn.modules.activation.__all__


class GeneBase(abc.ABC):
    """Base class for all Genes to inherit from"""

    def __init__(self, exons):
        if not isinstance(exons, collections.Mapping):
            raise TypeError(
                f"Exons must be a Mapping type, not {type(exons).__name__}.")
        self.exons = dict(exons)

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

    @abstractmethod
    def express(self):
        """
        Express the gene as a torch.nn.Module.
        Every Gene object must implement this method.
        """
        raise NotImplementedError(
            "Express function must be implemented!"
        )

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

    def express(self) -> nn.Sequential:
        cell = nn.Sequential(nn.Linear(self.in_features, self.out_features))
        if self.dropout:
            cell.add_module("dropout", nn.Dropout(p=0.5))
        if self.activation is not None:
            cell.add_module(self.activation.__class__.__name__,
                            getattr(nn, self.activation)())
        return cell

    @staticmethod
    def _validate_feature(name: str, value: int, allowed_range: set) -> int:
        return GeneBase._validate_feature(name, value, allowed_range)


class ConvBlock2dGene(GeneBase):
    allowed_channels = set(2 ** i for i in range(1, 6))
    allowed_kernel_size = set(range(2, 16, 2))

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int | tuple, activation: str = "ReLU",
                 batch_norm: bool | None = True) -> None:
        if activation not in _allowed_activations:
            raise ValueError(f"Unknown activation {activation}")

        in_channels = self._validate_feature("in_channels", in_channels,
                                             self.__class__.allowed_channels)
        out_channels = self._validate_feature("out_channels", out_channels,
                                              self.__class__.allowed_channels)
        kernel_size = self._validate_feature("kernel_size", kernel_size,
                                             self.__class__.allowed_kernel_size)

        exons = {"in_channels": in_channels,
                 "out_channels": out_channels,
                 "kernel_size": kernel_size,
                 "activation": activation,
                 "batch_norm": batch_norm}

        super().__init__(exons)

    def mutate(self, fnc):
        super().mutate(fnc)

    @staticmethod
    def _validate_feature(name: str, value: int, allowed_range: set) -> int:
        return super()._validate_feature(name, value, allowed_range)


class MaxPool2dGene(GeneBase):
    allowed_values = set(range(2, 10))

    def __init__(self, kernel_size, stride):
        kernel_size = self._validate_feature("kernel_size", kernel_size,
                                             self.__class__.allowed_values)

        exons = {"kernel_size": kernel_size, "stride": stride}
        super().__init__(exons)

    def mutate(self, fnc):
        super().mutate(fnc)

    @staticmethod
    def _validate_feature(name: str, value: int, allowed_range: set) -> int:
        return super()._validate_feature(name, value, allowed_range)


class SpatialAttentionGene(GeneBase):
    allowed_values = set(range(2, 10))
    in_channels = 2
    out_channels = 1
    activation = "Sigmoid"
    batch_norm = False

    def __init__(self, kernel_size: int | tuple):
        self.kernel_size = self._validate_feature("kernel_size", kernel_size,
                                                  self.__class__.allowed_values)

        self.conv_gene = ConvBlock2dGene(self.__class__.in_channels,
                                         self.__class__.out_channels,
                                         self.kernel_size, "Sigmoid", False)
        self.exons = {"kernel_size": self.kernel_size}
        super().__init__(self.exons)

    def mutate(self, *args, **kwargs) -> None:
        dk = 1 if random.random() < 0.5 else -1
        self.exons["kernel_size"] = self._validate_feature(
            "kernel_size",
            self.exons["kernel_size"] + dk,
            self.__class__.allowed_values)

    @staticmethod
    def _validate_feature(name: str, value: int, allowed_range: set) -> int:
        return super()._validate_feature(name, value, allowed_range)


class ChannelAttentionGene(GeneBase):
    allowed_range = set(range(2, 10))

    def __init__(self, se_ratio: int):
        self.se_ratio = self._validate_feature("se_ratio", se_ratio,
                                               self.__class__.allowed_range)
        super().__init__({"se_ratio": self.se_ratio})

    @staticmethod
    def _validate_feature(name: str, value: int, allowed_range: set) -> int:
        return super().validate(name, value, allowed_range)

    def mutate(self, *args, **kwargs) -> None:
        dk = 1 if random.random() < 0.5 else -1
        self.exons["kernel_size"] = self._validate_feature(
            "kernel_size",
            self.exons["kernel_size"] + dk,
            self.__class__.allowed_values)


class CBAMGene(GeneBase):
    def __init__(self,
                 ca_gene: ChannelAttentionGene,
                 sa_gene: SpatialAttentionGene,
                 ) -> None:
        self.ca_gene = ca_gene
        self.sa_gene = sa_gene
        super().__init__({
            "ca_gene": self.ca_gene.exons,
            "sa_gene": self.sa_gene.exons})

    def mutate(self, *args, **kwargs) -> None:
        self.ca_gene.mutate(*args, **kwargs)
        self.sa_gene.mutate(*args, **kwargs)
        self.exons["ca_gene"] = self.ca_gene.exons
        self.exons["sa_gene"] = self.sa_gene.exons


class FlattenGene(GeneBase):
    def __init__(self):
        exons = {"start_dim": 1, "end_dim": -1}
        super().__init__(exons)

    def mutate(self, *args, **kwargs) -> None: pass

    @staticmethod
    def _validate_feature(**kwargs) -> int: pass


def _build_layer(gene: GeneBase) -> nn.Module:
    name = gene.__class__.__name__.replace("Gene", "")
    if hasattr(globals(), name):
        module = globals()
    elif hasattr(nn, name):
        module = nn
    else:
        raise AttributeError(
            f"No class {name} found in either torch.nn or the current module")

    sig = inspect.signature(getattr(module, name))
    params = [p for p in gene.exons.keys() if p in sig.parameters]
    kwargs = {p: gene.exons[p] for p in params}
    return getattr(module, name)(**kwargs)
