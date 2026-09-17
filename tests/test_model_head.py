import torch
from torch import nn

from dnasty.defaults import S_2RB2D2
from dnasty.search_space.cbam import Genome, is_genome_valid
from dnasty.search_space.common import ConvBlock2d, LinearBlock


def _has_softmax(module: nn.Module) -> bool:
    return any(isinstance(m, (nn.Softmax, nn.LogSoftmax)) for m in module.modules())


def test_genome_outputs_logits(tiny_config):
    cfg = tiny_config.search_space.cbam
    genome = Genome.from_random(cfg, num_classes=9)
    while not is_genome_valid(genome, cfg):
        genome = Genome.from_random(cfg, num_classes=9)
    model = genome.to_module()
    assert not _has_softmax(model)
    last = genome.genes[next(reversed(genome.genes))]
    assert last.activation is None and last.dropout is False
    out = model(torch.randn(2, 1, 128, 128))
    assert out.shape == (2, 9)
    assert not torch.allclose(out.sum(dim=1), torch.ones(2))  # not a distribution


def test_reference_model_builds_and_outputs_logits():
    model = S_2RB2D2()
    assert not _has_softmax(model)
    assert model(torch.randn(2, 1, 128, 128)).shape == (2, 9)


def test_conv_block_order_and_batch_norm_flag():
    block = ConvBlock2d(1, 4, 3, activation="ReLU", batch_norm=True)
    assert [type(m) for m in block] == [nn.Conv2d, nn.BatchNorm2d, nn.ReLU]
    assert block[0].bias is None
    block = ConvBlock2d(1, 4, 3, activation="ReLU", batch_norm=False)
    assert [type(m) for m in block] == [nn.Conv2d, nn.ReLU]
    assert block[0].bias is not None


def test_linear_block_without_activation():
    block = LinearBlock(4, 2, dropout=False, activation=None)
    assert [type(m) for m in block] == [nn.Linear]
