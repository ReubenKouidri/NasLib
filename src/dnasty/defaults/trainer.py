from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dnasty.data.splits import DataModule
from dnasty.utils import Config, get_num_correct


@dataclass
class EpochResult:
    train_loss: float
    val_loss: float
    val_acc: float


@dataclass
class FitResult:
    best_val_acc: float
    epochs: list[EpochResult] = field(default_factory=list)


class Trainer:
    """Trains a model on a ``DataModule`` and reports validation accuracy.

    The same ``DataModule`` (hence the same split) must be shared by every
    trainer in a search so that fitness values are comparable.
    """

    def __init__(self, config: Config, datamodule: DataModule) -> None:
        self.train_cfg = config.train
        self.device = torch.device(config.get("device", "cpu"))
        self.datamodule = datamodule

    def _forward_pass(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer | None,
        criterion: nn.Module,
    ) -> tuple[float, float]:
        model.train(optimizer is not None)
        total_loss, correct, num_samples = 0.0, 0, 0
        for imgs, tgts in dataloader:
            imgs = imgs.to(self.device, non_blocking=True)
            tgts = tgts.to(self.device, non_blocking=True)
            preds = model(imgs)
            loss = criterion(preds, tgts)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            correct += get_num_correct(preds, tgts)
            total_loss += loss.item() * tgts.size(0)
            num_samples += tgts.size(0)
        if num_samples == 0:
            return 0.0, 0.0
        return total_loss / num_samples, correct / num_samples

    def _train(self, model, optimizer, criterion) -> float:
        loss, _ = self._forward_pass(
            model, self.datamodule.train_loader, optimizer, criterion
        )
        return loss

    @torch.inference_mode()
    def _eval(self, model, criterion) -> tuple[float, float]:
        return self._forward_pass(model, self.datamodule.val_loader, None, criterion)

    def fit(self, model: nn.Module, epochs: int) -> float:
        """Train for ``epochs`` and return the best validation accuracy."""
        return self.fit_detailed(model, epochs).best_val_acc

    def fit_detailed(self, model: nn.Module, epochs: int) -> FitResult:
        model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.train_cfg.lr)
        criterion = nn.CrossEntropyLoss()
        result = FitResult(best_val_acc=float("-inf"))
        for _ in range(epochs):
            train_loss = self._train(model, optimizer, criterion)
            val_loss, val_acc = self._eval(model, criterion)
            result.epochs.append(EpochResult(train_loss, val_loss, val_acc))
            result.best_val_acc = max(result.best_val_acc, val_acc)
        return result
