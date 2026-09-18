from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, random_split

from dnasty.utils import Config

DATA_DIR_ENV = "DNASTY_DATA_DIR"


def resolve_data_dir(config: Config, override: str | Path | None = None) -> Path:
    """Data root: CLI override > ``DNASTY_DATA_DIR`` > ``config.data_dir``."""
    if override is not None:
        return Path(override)
    env = os.environ.get(DATA_DIR_ENV)
    if env:
        return Path(env)
    return Path(config.get("data_dir", "data"))


def build_dataset(config: Config, data_dir: str | Path | None = None) -> Dataset:
    """Instantiate the dataset named in ``config.data``."""
    root = resolve_data_dir(config, data_dir)
    data_cfg = config.data
    if data_cfg.name == "cpsc2d":
        from dnasty.data.cpsc import CPSCDataset2D

        return CPSCDataset2D(
            data_dir=root / data_cfg.subdir,
            reference_path=root / data_cfg.reference,
            wavelet=data_cfg.get("wavelet", "mexh"),
            lead=data_cfg.get("lead", 3),
        )
    if data_cfg.name == "cpsc1d":
        from dnasty.data.cpsc import CPSCDataset

        return CPSCDataset(
            data_dir=root / data_cfg.subdir,
            reference_path=root / data_cfg.reference,
            lead=data_cfg.get("lead", 3),
        )
    raise ValueError(f"Unknown dataset '{data_cfg.name}'")


class DataModule:
    """One seeded train/validation split with its loaders.

    Both the search-time estimator and any post-search training must use the
    same ``DataModule`` so that fitness and final score are measured on the
    same validation records.
    """

    def __init__(
        self,
        dataset: Dataset,
        train_pct: float = 0.6,
        seed: int = 0,
        train_batch_size: int = 32,
        eval_batch_size: int = 64,
        num_workers: int = 0,
    ) -> None:
        total = len(dataset)  # type: ignore[arg-type]
        train_size = int(total * train_pct)
        generator = torch.Generator().manual_seed(seed)
        self.trainset, self.valset = random_split(
            dataset, [train_size, total - train_size], generator=generator
        )
        self.train_loader = DataLoader(
            self.trainset,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=num_workers,
            generator=torch.Generator().manual_seed(seed),
        )
        self.val_loader = DataLoader(
            self.valset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

    @classmethod
    def from_config(
        cls, config: Config, data_dir: str | Path | None = None
    ) -> DataModule:
        dataset = build_dataset(config, data_dir)
        return cls(
            dataset,
            train_pct=config.data.get("train_pct", 0.6),
            seed=config.get("seed", 0),
            train_batch_size=config.train.batch_size,
            eval_batch_size=config.eval.batch_size,
            num_workers=config.data.get("num_workers", 0),
        )
