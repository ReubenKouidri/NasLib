from __future__ import annotations

import torch.nn as nn


def get_activation(activation: str | nn.Module | None) -> nn.Module | None:
    """Resolve an activation spec to a module; ``None`` means no activation."""
    if activation is None:
        return None
    if isinstance(activation, str):
        return getattr(nn, activation)()
    if isinstance(activation, nn.Module):
        return activation
    raise TypeError("Activation must be a string, nn.Module, or None.")
