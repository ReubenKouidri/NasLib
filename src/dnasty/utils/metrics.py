"""Classification metrics for single-label and multi-label ECG tasks.

All functions accept NumPy arrays or torch tensors. Multi-label inputs are
``(N, C)``: ``scores`` are logits or probabilities (only their order per class
matters for AUROC), ``targets`` are 0/1.

- :func:`macro_auroc` is threshold-free and what the PTB-XL benchmark
  reports (Strodthoff et al. 2021).
- :func:`macro_f1` needs hard decisions and is what the CPSC 2018 challenge
  used.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from scipy.stats import rankdata

METRICS = ("accuracy", "macro_auroc", "macro_f1")


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


@torch.no_grad()
def get_num_correct(preds: torch.Tensor, tgts: torch.Tensor) -> int:
    """Number of samples whose arg-max prediction equals the target."""
    return int(preds.argmax(dim=1).eq(tgts).sum().item())


def accuracy(preds: Any, targets: Any) -> float:
    """Fraction of arg-max predictions equal to the integer targets."""
    preds = _to_numpy(preds)
    targets = _to_numpy(targets)
    if preds.ndim == 2:
        preds = preds.argmax(axis=1)
    return float(np.mean(preds == targets)) if targets.size else math.nan


def auroc(scores: Any, targets: Any) -> float:
    """Area under the ROC curve for one binary class.

    Computed from ranks (the Mann-Whitney U statistic), with tied scores
    given their average rank, which is what scikit-learn's trapezoidal
    ``roc_auc_score`` also yields. It equals the probability that a random
    positive outscores a random negative. Returns ``nan`` when the class has
    no positives or no negatives, since the curve is undefined then.
    """
    scores = _to_numpy(scores).astype(np.float64).ravel()
    targets = _to_numpy(targets).ravel().astype(bool)
    n_pos = int(targets.sum())
    n_neg = targets.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return math.nan
    ranks = rankdata(scores, method="average")
    rank_sum = float(ranks[targets].sum())
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def per_class_auroc(scores: Any, targets: Any) -> np.ndarray:
    """One-versus-rest AUROC per column; ``nan`` where undefined."""
    scores = _to_numpy(scores)
    targets = _to_numpy(targets)
    if scores.ndim != 2 or targets.shape != scores.shape:
        raise ValueError(
            f"Expected (N, C) scores and targets, got {scores.shape} and {targets.shape}"
        )
    return np.array(
        [auroc(scores[:, c], targets[:, c]) for c in range(scores.shape[1])]
    )


def macro_auroc(scores: Any, targets: Any) -> float:
    """Mean one-versus-rest AUROC over the classes for which it is defined.

    Classes with no positive or no negative example in ``targets`` are left
    out of the mean; the result is ``nan`` if that leaves no class.
    """
    values = per_class_auroc(scores, targets)
    valid = values[~np.isnan(values)]
    return float(valid.mean()) if valid.size else math.nan


def macro_f1(preds: Any, targets: Any, num_classes: int | None = None) -> float:
    """Mean F1 over classes, each class weighted equally.

    ``preds`` and ``targets`` are either ``(N, C)`` 0/1 multi-hot arrays or
    ``(N,)`` integer labels (then ``num_classes`` defaults to the largest
    label + 1). A class absent from both is skipped; a class present in only
    one of them scores 0.
    """
    preds = _to_numpy(preds)
    targets = _to_numpy(targets)
    if preds.ndim == 1:
        n = num_classes or int(max(preds.max(), targets.max())) + 1
        preds = np.eye(n, dtype=bool)[preds.astype(int)]
        targets = np.eye(n, dtype=bool)[targets.astype(int)]
    preds = preds.astype(bool)
    targets = targets.astype(bool)
    tp = (preds & targets).sum(axis=0)
    fp = (preds & ~targets).sum(axis=0)
    fn = (~preds & targets).sum(axis=0)
    denominator = 2 * tp + fp + fn
    present = denominator > 0
    if not present.any():
        return math.nan
    f1 = 2 * tp[present] / denominator[present]
    return float(f1.mean())


def compute_metric(name: str, scores: Any, targets: Any, multilabel: bool) -> float:
    """Evaluate ``name`` (one of :data:`METRICS`) on raw model outputs.

    For multi-label tasks ``scores`` are per-class logits and ``targets``
    multi-hot; F1 thresholds the logits at 0 (probability 0.5). For
    single-label tasks ``targets`` are class indices; AUROC uses softmax
    scores against one-hot targets and F1 uses the arg-max.
    """
    scores_t = torch.as_tensor(_to_numpy(scores))
    targets_t = torch.as_tensor(_to_numpy(targets))
    if name == "accuracy":
        if multilabel:
            raise ValueError("accuracy is undefined for multi-label targets")
        return accuracy(scores_t, targets_t)
    if not multilabel:
        onehot = torch.nn.functional.one_hot(targets_t.long(), scores_t.shape[1])
        if name == "macro_auroc":
            return macro_auroc(scores_t.softmax(dim=1), onehot)
        if name == "macro_f1":
            return macro_f1(scores_t.argmax(dim=1), targets_t, scores_t.shape[1])
    else:
        if name == "macro_auroc":
            return macro_auroc(scores_t, targets_t)
        if name == "macro_f1":
            return macro_f1(scores_t > 0, targets_t)
    raise ValueError(f"Unknown metric '{name}'; choose from {METRICS}")
