from .utils import get_activation

from .components.components import (
    LinearBlock,
    ConvBlock2d,
    Flatten
)

from .genetics.genes import (
    GeneBase,
    FlattenGene,
    ConvBlock2dGene,
    LinearBlockGene,
    MaxPool2dGene,
    create_conv_block_sequence,
    validate_feature
)

# TODO: Genome base/metaclass to inherit from.
#   currently Genome is in cbam space
# from .genetics.genome import (
#     Genome,
#     is_genome_valid,
#     adjust_linear_genes
# )

__all__ = [
    "GeneBase",
    "FlattenGene",
    "ConvBlock2dGene",
    "LinearBlockGene",
    "MaxPool2dGene",
    "LinearBlock",
    "ConvBlock2d",
    "Flatten",
    "create_conv_block_sequence",
    "validate_feature"
]

