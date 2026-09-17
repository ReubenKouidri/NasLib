"""Type aliases for size-like arguments (mirrors ``torch.nn.common_types``)."""

from __future__ import annotations

from typing import TypeVar

import torch.nn as nn

__all__ = [
    "act_t",
    "size_1_t",
    "size_2_opt_t",
    "size_2_t",
    "size_any_opt_t",
    "size_any_t",
]

T = TypeVar("T")

# Size parameters (kernel size, padding, ...): scalar or tuple.
size_any_t = int | tuple[int, ...]
size_1_t = int | tuple[int]
size_2_t = int | tuple[int, int]

# Optional size parameters (stride defaulting to kernel size, ...).
size_any_opt_t = int | None | tuple[int | None, ...]
size_2_opt_t = int | None | tuple[int | None, int | None]

act_t = str | nn.Module | None
