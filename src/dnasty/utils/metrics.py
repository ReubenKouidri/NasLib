from __future__ import annotations

import torch


@torch.no_grad()
def get_num_correct(preds: torch.Tensor, tgts: torch.Tensor) -> int:
    """Number of samples whose arg-max prediction equals the target."""
    return int(preds.argmax(dim=1).eq(tgts).sum().item())
