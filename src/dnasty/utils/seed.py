from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and torch RNGs.

    Genes are sampled with the ``random`` module, so this is what makes a
    search reproducible for a given config.
    """
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - legacy global RNG used by scipy/skimage
    torch.manual_seed(seed)
