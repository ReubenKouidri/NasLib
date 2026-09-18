"""dnasty: evolutionary neural architecture search for ECG classification."""

from importlib.metadata import PackageNotFoundError, version

from dnasty.data import DataModule
from dnasty.defaults import Trainer
from dnasty.estimators import (
    CachedEstimator,
    LowFidelityEstimator,
    MockEstimator,
    SyntheticEstimator,
)
from dnasty.search_space.cbam import Genome
from dnasty.search_strategies import RandomSearch, RegularizedEvolution, build_strategy
from dnasty.utils import Config, seed_everything

try:
    __version__ = version("dnasty")
except PackageNotFoundError:  # pragma: no cover - source checkout without install
    __version__ = "0.0.0"

__all__ = [
    "CachedEstimator",
    "Config",
    "DataModule",
    "Genome",
    "LowFidelityEstimator",
    "MockEstimator",
    "RandomSearch",
    "RegularizedEvolution",
    "SyntheticEstimator",
    "Trainer",
    "__version__",
    "build_strategy",
    "seed_everything",
]
