from __future__ import annotations

import abc
import hashlib
import math
import random
import statistics

from dnasty.data.splits import DataModule
from dnasty.defaults.trainer import Trainer
from dnasty.search_space.cbam import ConvBlock2dGene, Genome
from dnasty.utils import Config


class Estimator(abc.ABC):
    """Maps a genome to a fitness estimate (higher is better)."""

    @abc.abstractmethod
    def fit(self, genome: Genome) -> float: ...


class LowFidelityEstimator(Estimator):
    """Train each genome for a few epochs (``estimator.low_fidelity.epochs``)
    and use the best validation accuracy as its fitness."""

    def __init__(
        self,
        config: Config,
        datamodule: DataModule | None = None,
        trainer: Trainer | None = None,
    ) -> None:
        est_cfg = config.estimator
        section = est_cfg.get("low_fidelity") or est_cfg.get("early_stopping")
        self.fidelity = section.get("epochs", section.get("fidelity", 1))
        if trainer is None:
            if datamodule is None:
                datamodule = DataModule.from_config(config)
            trainer = Trainer(config, datamodule)
        self.trainer = trainer

    def fit(self, genome: Genome) -> float:
        return self.trainer.fit(genome.to_module(), self.fidelity)


# Historical name; it never implemented early stopping.
EarlyStoppingEstimator = LowFidelityEstimator


class MockEstimator(Estimator):
    """Random fitness in [0.5, 1); builds the module to check it expresses.

    Useless for comparing strategies (every architecture is equally good in
    expectation); use :class:`SyntheticEstimator` for that.
    """

    def __init__(self, seed: int = 0, build_module: bool = True) -> None:
        self.rng = random.Random(seed)
        self.build_module = build_module

    def fit(self, genome: Genome) -> float:
        if self.build_module:
            genome.to_module()
        return self.rng.uniform(0.5, 1.0)


class SyntheticEstimator(Estimator):
    """Deterministic, training-free fitness landscape for benchmarking strategies.

    A toy stand-in for a NAS benchmark: the fitness of an architecture is a
    fixed function of its structure plus a small noise term that is seeded by
    the architecture itself, so re-evaluating the same genome gives the same
    number. The landscape rewards

    - four conv blocks in total (``depth``),
    - kernel sizes near 5 (``kernel``),
    - 64 channels per conv block (``width``),
    - few parameters, ``exp(-params / 500k)`` (``size``),

    weighted 0.35 / 0.25 / 0.20 / 0.20, and clips to [0, 1]. It says nothing
    about real accuracy; it only checks that a strategy climbs a landscape
    that has structure, which random noise (``MockEstimator``) cannot.
    """

    def __init__(self, seed: int = 0, noise: float = 0.02) -> None:
        self.seed = seed
        self.noise = noise

    @staticmethod
    def _kernel(gene: ConvBlock2dGene) -> int:
        k = gene.kernel_size
        return int(k if isinstance(k, int) else k[0])

    def score(self, genome: Genome) -> float:
        """Noise-free score."""
        convs = [g for g in genome.genes.values() if isinstance(g, ConvBlock2dGene)]
        depth = 1 - min(abs(len(convs) - 4), 4) / 4
        kernel = statistics.fmean(
            1 - min(abs(self._kernel(g) - 5), 11) / 11 for g in convs
        )
        width = statistics.fmean(
            1 - abs(math.log2(max(int(g.out_channels), 1)) - 6) / 6 for g in convs
        )
        size = math.exp(-genome.num_params / 500_000)
        return 0.35 * depth + 0.25 * kernel + 0.20 * width + 0.20 * size

    def fit(self, genome: Genome) -> float:
        digest = hashlib.sha256(
            repr((self.seed, genome.arch_key())).encode()
        ).hexdigest()
        rng = random.Random(int(digest[:16], 16))
        value = self.score(genome) + rng.gauss(0.0, self.noise)
        return min(1.0, max(0.0, value))


class CachedEstimator(Estimator):
    """Memoise an estimator on :meth:`Genome.arch_key`.

    Evolution revisits architectures; with a training estimator that is wasted
    compute, and for a benchmark it also makes the fitness of a given
    architecture a fixed number, as in tabular NAS benchmarks.
    """

    def __init__(self, inner: Estimator) -> None:
        self.inner = inner
        self.cache: dict[tuple, float] = {}
        self.hits = 0
        self.misses = 0

    def fit(self, genome: Genome) -> float:
        key = genome.arch_key()
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        value = float(self.inner.fit(genome))
        self.cache[key] = value
        return value
