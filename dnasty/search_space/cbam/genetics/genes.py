import random
from dnasty.my_utils import Config
from dnasty.my_utils.types import *
from dnasty.search_space.common import GeneBase, FlattenGene, LinearBlockGene
from dnasty.search_space.common import create_conv_block_sequence
from dnasty.search_space.common import validate_feature


def create_gene_sequence(cfg: Config) -> list:
    genes = []
    for _ in range(random.randint(1, cfg.cells)):
        genes.extend(create_conv_block_sequence(cfg))
        genes.append(CBAMGene.from_random())
    genes.append(FlattenGene())
    genes.extend(LinearBlockGene.from_random() for _ in range(cfg.linear))
    return genes


class SpatialAttentionGene(GeneBase):
    """
    Gene encoding a spatial attention module: https://arxiv.org/abs/1807.06521

    Attributes:
        _feature_ranges (dict[set]): Defines the allowed range for the kernel
                                     size of the spatial attention mechanism.

    Args:
        kernel_size (size_2_t): The size of the kernel to be used.
         Can be a single integer or a tuple of two integers.
    """
    _feature_ranges = {"kernel_size": set(range(2, 17))}

    def __init__(self, kernel_size: size_2_t):
        kernel_size = validate_feature(
            "kernel_size", kernel_size,
            self._feature_ranges["kernel_size"])

        super().__init__({"kernel_size": kernel_size})

    def mutate(self) -> None:
        dk = 1 if random.random() < 0.5 else -1
        self.kernel_size += dk

    @property
    def num_params(self):
        if isinstance(self.kernel_size, int):
            return self.kernel_size ** 2 + 1
        return self.kernel_size[0] * self.kernel_size[1] + 1


class ChannelAttentionGene(GeneBase):
    """
    Gene encoding a channel attention module: https://arxiv.org/abs/1807.06521

    Attributes:
        _feature_ranges (dict): Defines the allowed range for the
        squeeze-excitation ratio

    Args:
        se_ratio: the squeeze-excitation ratio.
        in_channels: Note that this is not strictly an independent var as
        it is set to match the output of the previous layer.
    """
    _feature_ranges = {"se_ratio": set(range(2, 17))}

    def __init__(self, se_ratio: int, in_channels: int = 1):
        se_ratio = validate_feature(
            "se_ratio", se_ratio,
            ChannelAttentionGene._feature_ranges["se_ratio"])

        super().__init__({"in_channels": in_channels, "se_ratio": se_ratio})

    def mutate(self) -> None:
        dr = 1 if random.random() < 0.5 else -1
        self.se_ratio += dr

    @property
    def num_params(self):
        if self.in_channels // self.se_ratio == 0:
            return 2 * self.in_channels
        return 2 * self.in_channels * (self.in_channels // self.se_ratio)


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
        the moment so that these genetics can be activated later.
        This might change.
        spatial_gene (SpatialAttentionGene): Same as above...
    """

    @classmethod
    def from_random(cls):
        """
        Since the CBAMGene follows a different pattern from other genes,
        we override the from_random method to create CBAMGene instances
        with randomly initialized channel and spatial genes.
        """
        channel_gene = ChannelAttentionGene.from_random()
        spatial_gene = SpatialAttentionGene.from_random()
        return cls(channel_gene=channel_gene, spatial_gene=spatial_gene)

    @property
    def out_channels(self):
        return self.in_channels

    def __init__(self,
                 channel_gene: ChannelAttentionGene,
                 spatial_gene: SpatialAttentionGene):
        """
        Bypass redundant validation call to GeneBase as sub-genes are already
        validated.
        """
        object.__setattr__(self, "channel_gene", channel_gene)
        object.__setattr__(self, "spatial_gene", spatial_gene)
        super().__init__({"in_channels": channel_gene.in_channels,
                          "se_ratio": channel_gene.se_ratio,
                          "kernel_size": spatial_gene.kernel_size})

    def mutate(self, *args, **kwargs) -> None:
        self.channel_gene.mutate(*args, **kwargs)
        self.spatial_gene.mutate(*args, **kwargs)

    def sync(self):
        self.channel_gene.in_channels = self.in_channels
        self.channel_gene.se_ratio = self.se_ratio
        self.spatial_gene.kernel_size = self.kernel_size

    @property
    def num_params(self):
        return self.channel_gene.num_params + self.spatial_gene.num_params
