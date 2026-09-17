"""CPSC 2018 (China Physiological Signal Challenge) single-lead datasets.

Records are ``.mat`` files with ``ECG.data`` of shape (12, n_samples) at
500 Hz; labels come from a reference CSV with up to three diagnoses per
recording (1-based class ids). Labels are joined on the recording id, so the
data directory may hold any subset of the reference file's recordings.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat
from scipy.ndimage import uniform_filter1d
from torch import Tensor
from torch.utils.data import Dataset

from dnasty.utils import wavelets as wavelets_module

MISSING_LABEL = -1


def read_reference(reference_path: str | Path) -> dict[str, tuple[int, int, int]]:
    """Map recording id -> (label1, label2, label3), 0-based, -1 if absent."""
    targets: dict[str, tuple[int, int, int]] = {}
    with Path(reference_path).open(newline="") as ref:
        reader = csv.reader(ref)
        next(reader)  # header
        for row in reader:
            labels = [int(v) - 1 if v.strip() else MISSING_LABEL for v in row[1:4]]
            labels += [MISSING_LABEL] * (3 - len(labels))
            targets[row[0]] = (labels[0], labels[1], labels[2])
    return targets


class CPSCDataset(Dataset):
    AR_classes = {
        "SR": 0,
        "AF": 1,
        "I-AVB": 2,
        "LBBB": 3,
        "RBBB": 4,
        "PAC": 5,
        "PVC": 6,
        "STD": 7,
        "STE": 8,
    }
    sampling_rate = 500
    length = 4 * sampling_rate  # first 4 s

    def __init__(
        self,
        data_dir: str | Path,
        reference_path: str | Path,
        normalize: bool = True,
        smoothen: bool = True,
        trim: bool = True,
        lead: int = 3,
        test: bool = False,
        load_in_memory: bool = True,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.test = test
        self.normalize = normalize
        self.trim = trim
        self.smoothen = smoothen
        self.lead = lead - 1  # leads are numbered 1..12
        self.load_in_memory = load_in_memory

        reference = read_reference(reference_path)
        self.filenames = sorted(p.name for p in self.data_dir.glob("*.mat"))
        self.record_ids = [Path(f).stem for f in self.filenames]
        missing = [r for r in self.record_ids if r not in reference]
        if missing:
            raise ValueError(
                f"{len(missing)} recordings have no label in {reference_path}: "
                f"{missing[:5]}..."
            )
        self.targets = torch.tensor(
            [reference[r] for r in self.record_ids], dtype=torch.int64
        )

        if self.load_in_memory:
            self.data = torch.stack([self._load(f) for f in self.filenames])

    def _load(self, filename: str) -> Tensor:
        mat = loadmat(self.data_dir / filename)
        ecg = mat["ECG"]["data"][0][0][self.lead]
        return torch.as_tensor(self._process_data(ecg), dtype=torch.float32)

    def _process_data(self, data: np.ndarray) -> np.ndarray:
        if self.trim:
            data = self._trim_data(data, self.length, step=2)
        if self.normalize:
            data = self._normalize(data)
        if self.smoothen:
            data = self._smoothen(data)
        return data

    @staticmethod
    def _normalize(data: np.ndarray) -> np.ndarray:
        lo, hi = np.min(data), np.max(data)
        return (data - lo) / (hi - lo) if hi > lo else np.zeros_like(data)

    @staticmethod
    def _trim_data(data: np.ndarray, size: int, step: int = 1) -> np.ndarray:
        return data[:size:step]

    @staticmethod
    def _smoothen(data: np.ndarray, window: int = 8) -> np.ndarray:
        return uniform_filter1d(data, size=window, mode="nearest")

    def _target(self, item: int) -> Tensor:
        return self.targets[item] if self.test else self.targets[item][0]

    def __getitem__(self, item: int) -> tuple[Tensor, Tensor]:
        if self.load_in_memory:
            return self.data[item], self._target(item)
        return self._load(self.filenames[item]), self._target(item)

    def __len__(self) -> int:
        return len(self.filenames)


class CPSCDataset2D(CPSCDataset):
    """CPSC records as single-channel 128x128 wavelet-scalogram images."""

    wavelets = {"mexh": 64, "cmor": 64}  # max widths of the CWT

    def __init__(
        self,
        data_dir: str | Path,
        reference_path: str | Path,
        wavelet: str = "mexh",
        lead: int = 3,
        load_in_memory: bool = True,
        test: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            data_dir=data_dir,
            reference_path=reference_path,
            load_in_memory=load_in_memory,
            lead=lead,
            test=test,
            **kwargs,
        )
        self.wavelet = wavelet if wavelet in self.wavelets else "mexh"
        self.wavelet_fnc = getattr(wavelets_module, self.wavelet)
        if self.load_in_memory:
            self.images = torch.stack([self._to_image(ecg) for ecg in self.data])

    def _to_image(self, ecg: Tensor) -> Tensor:
        img = self.wavelet_fnc(ecg.numpy(), self.wavelets[self.wavelet])
        return torch.as_tensor(img, dtype=torch.float32).unsqueeze(0)

    def __getitem__(self, item: int) -> tuple[Tensor, Tensor]:
        if self.load_in_memory:
            return self.images[item], self._target(item)
        ecg, target = super().__getitem__(item)
        return self._to_image(ecg), target
