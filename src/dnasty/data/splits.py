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
    """Instantiate the dataset named in ``config.data`` (random-split datasets)."""
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


def build_ptbxl_splits(
    config: Config, data_dir: str | Path | None = None
) -> tuple[Dataset, Dataset]:
    """Train and validation ``PTBXLDataset`` from ``config.data`` folds.

    Config keys (under ``data``): ``subdir`` (default ``ptbxl``), ``task``,
    ``sampling_rate``, ``train_folds`` (default 1-8), ``val_fold`` (default
    9), ``leads`` (list or ``all``), ``normalize``, ``min_likelihood``.
    """
    from dnasty.data.ptbxl import PTBXLDataset

    root = resolve_data_dir(config, data_dir) / config.data.get("subdir", "ptbxl")
    data_cfg = config.data
    leads = data_cfg.get("leads", "all")
    common = dict(
        root=root,
        task=data_cfg.get("task", "superdiagnostic"),
        sampling_rate=int(data_cfg.get("sampling_rate", 100)),
        leads=None if leads in (None, "all") else list(leads),
        normalize=data_cfg.get("normalize", "record"),
        min_likelihood=float(data_cfg.get("min_likelihood", 0.0)),
    )
    train_folds = data_cfg.get("train_folds", list(range(1, 9)))
    val_folds = data_cfg.get("val_folds", [data_cfg.get("val_fold", 9)])
    train = PTBXLDataset(folds=list(train_folds), **common)
    val = PTBXLDataset(folds=list(val_folds), **common)
    return train, val


def _is_multilabel(dataset: Dataset) -> bool:
    if hasattr(dataset, "multilabel"):
        return bool(dataset.multilabel)
    if len(dataset) == 0:  # type: ignore[arg-type]
        return False
    target = dataset[0][1]
    target = torch.as_tensor(target)
    return bool(target.is_floating_point() and target.dim() == 1 and target.numel() > 1)


class DataModule:
    """One train/validation split with its loaders.

    Either pass ``dataset`` for a seeded random split (``train_pct``), or
    ``train_set`` and ``val_set`` for predefined splits such as PTB-XL's
    folds. Both the search-time estimator and any post-search training must
    use the same ``DataModule`` so that fitness and final score are measured
    on the same validation records.

    Attributes:
        multilabel: targets are multi-hot float vectors (``BCEWithLogitsLoss``
            and macro-AUROC) rather than class indices.
        num_classes, in_channels: taken from the dataset when it exposes
            them, else ``None``.
    """

    def __init__(
        self,
        dataset: Dataset | None = None,
        train_pct: float = 0.6,
        seed: int = 0,
        train_batch_size: int = 32,
        eval_batch_size: int = 64,
        num_workers: int = 0,
        *,
        train_set: Dataset | None = None,
        val_set: Dataset | None = None,
    ) -> None:
        self.trainset: Dataset
        self.valset: Dataset
        if dataset is not None:
            if train_set is not None or val_set is not None:
                raise ValueError("Pass either dataset or train_set/val_set, not both")
            total = len(dataset)  # type: ignore[arg-type]
            train_size = int(total * train_pct)
            generator = torch.Generator().manual_seed(seed)
            self.trainset, self.valset = random_split(
                dataset, [train_size, total - train_size], generator=generator
            )
            source: Dataset = dataset
        elif train_set is not None and val_set is not None:
            self.trainset, self.valset = train_set, val_set
            source = train_set
        else:
            raise ValueError("Pass dataset or both train_set and val_set")

        self.multilabel = _is_multilabel(source)
        self.num_classes: int | None = getattr(source, "num_classes", None)
        self.in_channels: int | None = getattr(source, "in_channels", None)
        self.train_loader = DataLoader(
            self.trainset,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=num_workers,
            generator=torch.Generator().manual_seed(seed),
            persistent_workers=num_workers > 0,
        )
        self.val_loader = DataLoader(
            self.valset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
        )

    @classmethod
    def from_config(
        cls, config: Config, data_dir: str | Path | None = None
    ) -> DataModule:
        loader_kwargs = dict(
            seed=config.get("seed", 0),
            train_batch_size=config.train.batch_size,
            eval_batch_size=config.eval.batch_size,
            num_workers=config.data.get("num_workers", 0),
        )
        if config.data.name == "ptbxl":
            train, val = build_ptbxl_splits(config, data_dir)
            return cls(train_set=train, val_set=val, **loader_kwargs)
        dataset = build_dataset(config, data_dir)
        return cls(
            dataset, train_pct=config.data.get("train_pct", 0.6), **loader_kwargs
        )
