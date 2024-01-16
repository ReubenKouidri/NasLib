from __future__ import annotations

import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from dnasty.my_utils import load_2d_dataset, get_num_correct
from dnasty.my_utils.config import Config

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def _to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return type(data)(_to_device(x, device) for x in data)
    return data.to(device, non_blocking=True)


class DeviceDataLoader:
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        for b in self.dl:
            yield _to_device(b, self.device)

    def __len__(self):
        return len(self.dl)


class Trainer:
    def __init__(self,
                 config: Config,
                 checkpoint_dir: str = 'checkpoints',
                 algo_mode: bool | None = False
                 ) -> None:
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        self.trainset, self.valset = self._load_data(self.config.data.data_path,
                                                     self.config.data.reference_path)
        self.algo_mode = algo_mode

    @staticmethod
    def _load_data(data_path, ref_path, train_pct: float = 0.8):
        dataset = load_2d_dataset(data_path, ref_path)
        total_size = len(dataset)
        train_size = int(total_size * train_pct)
        val_size = total_size - train_size
        train_dataset, val_dataset = random_split(dataset,
                                                  [train_size, val_size])
        return train_dataset, val_dataset

    def _forward_pass(self,
                      model: torch.nn.Module,
                      mode: str,
                      dataloader: torch.utils.data.DataLoader,
                      optimizer: torch.optim.Optimizer | None,
                      criterion: nn.Module
                      ) -> tuple[float, float]:
        model.train() if mode == "train" else model.validate()
        total_loss, correct = 0.0, 0
        for imgs, tgts in dataloader:
            imgs = imgs.to(self.config.device, non_blocking=True)
            tgts = tgts.to(self.config.device, non_blocking=True)
            preds = model(imgs)
            loss = criterion(preds, tgts)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            correct += get_num_correct(preds, tgts)
            total_loss += loss.item()

        num_samples = len(dataloader.dataset)
        return total_loss / num_samples, correct / num_samples

    def _train(self, model, trainloader, optimizer, criterion):
        return self._forward_pass(model, "train", trainloader, optimizer,
                                  criterion)

    @torch.inference_mode()
    def _eval(self, model, valloader, criterion) -> tuple[float, float]:
        return self._forward_pass(model, "val", valloader, None, criterion)

    @staticmethod
    def _ddl(dataloader, device):
        return DeviceDataLoader(dataloader, device)

    def fit(self, model):
        train_loader = self._ddl(DataLoader(self.trainset,
                                            self.config.train.batch_size,
                                            shuffle=True),
                                 self.config.device)
        val_loader = self._ddl(DataLoader(self.valset,
                                          self.config.validation.batch_size,
                                          shuffle=False),
                               self.config.device)

        optimizer = getattr(torch.optim, self.config.optimizer.name)(
            model.parameters(), **self.config.optimizer.kwargs)
        criterion = getattr(torch.nn, self.config.criterion.name)()

        highest_val_score = float('-inf')
        for epoch in range(self.config.train.epochs):
            self._train(model, train_loader, optimizer, criterion)
            val_loss, val_accuracy = self._eval(model, val_loader, criterion)
            highest_val_score = max(highest_val_score, val_accuracy)
            logging.info(
                f"Epoch {epoch + 1}/{self.config.train.epochs}: Validation "
                f"Loss = {val_loss:.4f}, Validation Accuracy = "
                f"{val_accuracy:.4f}")

        return highest_val_score
