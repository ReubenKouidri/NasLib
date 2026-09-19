from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dnasty.data.splits import DataModule
from dnasty.utils import Config
from dnasty.utils.metrics import METRICS, compute_metric

logger = logging.getLogger(__name__)


@dataclass
class EpochResult:
    train_loss: float
    val_loss: float
    val_score: float

    @property
    def val_acc(self) -> float:  # backwards-compatible name
        return self.val_score


@dataclass
class FitResult:
    metric: str
    best_val_score: float
    epochs: list[EpochResult] = field(default_factory=list)

    @property
    def best_val_acc(self) -> float:  # backwards-compatible name
        return self.best_val_score


class Trainer:
    """Trains a model on a ``DataModule`` and scores it on the validation set.

    The loss follows the targets: cross-entropy for class indices,
    ``BCEWithLogitsLoss`` for multi-hot vectors. The validation score is
    ``config.metric`` (``accuracy``, ``macro_auroc`` or ``macro_f1``), by
    default accuracy for single-label and macro-AUROC for multi-label data,
    computed once over the whole validation set from the raw logits.

    The same ``DataModule`` (hence the same split) must be shared by every
    trainer in a search so that fitness values are comparable.
    """

    def __init__(self, config: Config, datamodule: DataModule) -> None:
        self.train_cfg = config.train
        self.device = torch.device(config.get("device", "cpu"))
        self.datamodule = datamodule
        self.multilabel = datamodule.multilabel
        default = "macro_auroc" if self.multilabel else "accuracy"
        self.metric = str(config.get("metric", default))
        if self.metric not in METRICS:
            raise ValueError(f"Unknown metric '{self.metric}'; choose from {METRICS}")
        if self.metric == "accuracy" and self.multilabel:
            raise ValueError(
                "accuracy is undefined for multi-label data; use macro_auroc"
            )
        self.criterion: nn.Module = (
            nn.BCEWithLogitsLoss() if self.multilabel else nn.CrossEntropyLoss()
        )

    def _forward_pass(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer | None,
        criterion: nn.Module | None = None,
        collect: bool = False,
    ) -> tuple[float, torch.Tensor | None, torch.Tensor | None]:
        """Mean loss per sample and, if ``collect``, all logits and targets."""
        criterion = criterion or self.criterion
        model.train(optimizer is not None)
        total_loss, num_samples = 0.0, 0
        outputs: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        for inputs, tgts in dataloader:
            inputs = inputs.to(self.device, non_blocking=True)
            tgts = tgts.to(self.device, non_blocking=True)
            preds = model(inputs)
            loss = criterion(preds, tgts)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * tgts.size(0)
            num_samples += tgts.size(0)
            if collect:
                outputs.append(preds.detach().cpu())
                targets.append(tgts.detach().cpu())
        if num_samples == 0:
            return 0.0, None, None
        mean_loss = total_loss / num_samples
        if not collect:
            return mean_loss, None, None
        return mean_loss, torch.cat(outputs), torch.cat(targets)

    def _train(self, model, optimizer, criterion=None) -> float:
        loss, _, _ = self._forward_pass(
            model, self.datamodule.train_loader, optimizer, criterion
        )
        return loss

    @torch.inference_mode()
    def _eval(self, model, criterion=None) -> tuple[float, float]:
        """Validation loss and the configured metric."""
        loss, outputs, targets = self._forward_pass(
            model, self.datamodule.val_loader, None, criterion, collect=True
        )
        if outputs is None or targets is None:
            return loss, 0.0
        score = compute_metric(self.metric, outputs, targets, self.multilabel)
        if math.isnan(score):
            logger.warning(
                "%s is undefined on this validation set (a class has no positive "
                "or no negative example); scoring 0.0",
                self.metric,
            )
            score = 0.0
        return loss, score

    def fit(self, model: nn.Module, epochs: int) -> float:
        """Train for ``epochs`` and return the best validation score."""
        return self.fit_detailed(model, epochs).best_val_score

    def fit_detailed(self, model: nn.Module, epochs: int) -> FitResult:
        model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.train_cfg.lr)
        result = FitResult(metric=self.metric, best_val_score=float("-inf"))
        for _ in range(epochs):
            train_loss = self._train(model, optimizer)
            val_loss, val_score = self._eval(model)
            result.epochs.append(EpochResult(train_loss, val_loss, val_score))
            result.best_val_score = max(result.best_val_score, val_score)
        return result
