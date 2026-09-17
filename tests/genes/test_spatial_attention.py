import copy

import torch
from torch import nn

from dnasty.search_space.cbam import SpatialAttentionGene


def test_init_and_getattr():
    gene = SpatialAttentionGene(7)
    assert gene.kernel_size == 7
    assert len(gene) == 1
    try:
        _ = gene.invalid
    except AttributeError:
        pass
    else:
        raise AssertionError("expected AttributeError")


def test_express_gate_ends_with_sigmoid():
    module = SpatialAttentionGene(7).to_module()
    assert isinstance(module.conv[-1], nn.Sigmoid)
    x = torch.randn(16, 64, 32, 32)
    y = module(x)
    assert y.shape == x.shape


def test_deepcopy():
    gene = SpatialAttentionGene(7)
    copied = copy.deepcopy(gene)
    assert copied is not gene
    assert copied.__dict__ == gene.__dict__
