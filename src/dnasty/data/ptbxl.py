"""PTB-XL (Wagner et al. 2020): 21,837 12-lead 10 s ECGs with SCP statements.

Directory layout (as downloaded from PhysioNet, ``ptb-xl/1.0.3``)::

    <root>/ptbxl_database.csv      one row per record: ecg_id, scp_codes,
                                   strat_fold (1-10), filename_lr, filename_hr
    <root>/scp_statements.csv      one row per SCP code with its task flags
    <root>/records100/<group>/<id>_lr.{hea,dat}    100 Hz
    <root>/records500/<group>/<id>_hr.{hea,dat}    500 Hz

Tasks follow Strodthoff et al. (2021): ``superdiagnostic`` (5 classes),
``subdiagnostic`` (23), ``diagnostic`` (44), ``form`` (19), ``rhythm`` (12)
and ``all`` (71). Labels are multi-hot. Folds 1-8 are the recommended
training set, fold 9 validation and fold 10 test; folds are patient-disjoint
and folds 9 and 10 contain only high-quality labels.

Signals are read once with ``wfdb``, standardised per record and lead, and
stored as a float16 memmap under ``<root>/cache/`` so that later runs (and
``DataLoader`` worker processes) touch no WFDB files. Records listed in the
CSV but missing on disk are skipped, so a subset download works.
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

TASKS = ("all", "diagnostic", "subdiagnostic", "superdiagnostic", "form", "rhythm")
LEADS = ("I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6")
NORMALIZATIONS = ("record", "none")
CACHE_VERSION = 1


def load_metadata(root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The record table (indexed by ``ecg_id``) and the SCP statement table."""
    root = Path(root)
    database = pd.read_csv(root / "ptbxl_database.csv", index_col="ecg_id")
    database["scp_codes"] = database["scp_codes"].apply(ast.literal_eval)
    statements = pd.read_csv(root / "scp_statements.csv", index_col=0)
    return database, statements


def task_mapping(statements: pd.DataFrame, task: str) -> dict[str, str]:
    """SCP code -> class name for ``task`` (codes not in the task are absent)."""
    if task not in TASKS:
        raise ValueError(f"Unknown task '{task}'; choose from {TASKS}")
    if task == "all":
        return {code: code for code in statements.index}
    if task in ("diagnostic", "subdiagnostic", "superdiagnostic"):
        rows = statements[statements["diagnostic"] == 1.0]
        column = {
            "diagnostic": None,
            "subdiagnostic": "diagnostic_subclass",
            "superdiagnostic": "diagnostic_class",
        }[task]
        if column is None:
            return {code: code for code in rows.index}
        return {code: str(cls) for code, cls in rows[column].items() if pd.notna(cls)}
    rows = statements[statements[task] == 1.0]
    return {code: code for code in rows.index}


def aggregate_labels(
    scp_codes: dict[str, float], mapping: dict[str, str], min_likelihood: float = 0.0
) -> set[str]:
    """Class names of ``scp_codes`` under ``mapping``.

    ``scp_codes`` maps an SCP code to its likelihood in percent (0 means
    unknown); codes below ``min_likelihood`` are ignored. The PTB-XL
    benchmark code uses every listed code, i.e. ``min_likelihood=0``.
    """
    return {
        mapping[code]
        for code, likelihood in scp_codes.items()
        if code in mapping and likelihood >= min_likelihood
    }


def _cache_key(sampling_rate: int, normalize: str, ecg_ids: Sequence[int]) -> str:
    digest = hashlib.sha1(
        json.dumps([CACHE_VERSION, sampling_rate, normalize, list(ecg_ids)]).encode()
    ).hexdigest()[:12]
    return f"{sampling_rate}hz-{normalize}-{digest}"


class PTBXLCache:
    """Preprocessed signals for every available record, as a float16 memmap.

    ``signals.npy`` has shape ``(n_records, 12, length)`` in record order of
    ``meta.json["ecg_ids"]``. The key includes the record ids, so a cache is
    rebuilt when more records are downloaded.
    """

    def __init__(
        self,
        root: str | Path,
        database: pd.DataFrame,
        sampling_rate: int = 100,
        normalize: str = "record",
        cache_dir: str | Path | None = None,
    ) -> None:
        if sampling_rate not in (100, 500):
            raise ValueError("sampling_rate must be 100 or 500")
        if normalize not in NORMALIZATIONS:
            raise ValueError(f"normalize must be one of {NORMALIZATIONS}")
        self.root = Path(root)
        self.sampling_rate = sampling_rate
        self.normalize = normalize
        column = "filename_lr" if sampling_rate == 100 else "filename_hr"
        available = database[
            database[column].apply(lambda f: (self.root / f"{f}.dat").is_file())
        ]
        self.filenames = {int(i): str(f) for i, f in available[column].items()}
        self.ecg_ids = sorted(self.filenames)
        if not self.ecg_ids:
            raise FileNotFoundError(
                f"No PTB-XL {sampling_rate} Hz records found under {self.root}; "
                "run `dnasty data download ptbxl`"
            )
        key = _cache_key(sampling_rate, normalize, self.ecg_ids)
        self.dir = Path(cache_dir) if cache_dir else self.root / "cache" / key
        self.signals_path = self.dir / "signals.npy"
        self.meta_path = self.dir / "meta.json"
        self.row_of = {ecg_id: row for row, ecg_id in enumerate(self.ecg_ids)}
        self._signals: np.ndarray | None = None
        self.meta = self._load_meta() or self.build()

    def _load_meta(self) -> dict[str, Any] | None:
        if not (self.meta_path.is_file() and self.signals_path.is_file()):
            return None
        meta = json.loads(self.meta_path.read_text())
        if meta.get("ecg_ids") != self.ecg_ids or meta.get("version") != CACHE_VERSION:
            return None
        return meta

    @staticmethod
    def _read(path: Path) -> tuple[np.ndarray, list[str]]:
        try:
            import wfdb
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise ImportError(
                "wfdb is required to build the PTB-XL cache: pip install 'dnasty[data]'"
            ) from exc
        signal, fields = wfdb.rdsamp(str(path))
        return np.asarray(signal, dtype=np.float32).T, list(fields["sig_name"])

    def preprocess(self, signal: np.ndarray) -> np.ndarray:
        """``(leads, length)`` in mV -> standardised per lead (if enabled)."""
        if self.normalize == "record":
            mean = signal.mean(axis=1, keepdims=True)
            std = signal.std(axis=1, keepdims=True)
            signal = (signal - mean) / (std + 1e-6)
        return signal

    def build(self) -> dict[str, Any]:
        from tqdm import tqdm

        self.dir.mkdir(parents=True, exist_ok=True)
        first, sig_names = self._read(self.root / self.filenames[self.ecg_ids[0]])
        if [n.upper() for n in sig_names] != list(LEADS):
            raise ValueError(f"Unexpected lead order {sig_names}")
        n_leads, length = first.shape
        logger.info(
            "Building PTB-XL cache for %d records at %d Hz in %s",
            len(self.ecg_ids),
            self.sampling_rate,
            self.dir,
        )
        signals = np.lib.format.open_memmap(
            self.signals_path,
            mode="w+",
            dtype=np.float16,
            shape=(len(self.ecg_ids), n_leads, length),
        )
        for row, ecg_id in enumerate(
            tqdm(self.ecg_ids, desc="ptbxl cache", unit="rec")
        ):
            signal = (
                first if row == 0 else self._read(self.root / self.filenames[ecg_id])[0]
            )
            if signal.shape != (n_leads, length):
                raise ValueError(
                    f"Record {ecg_id} has shape {signal.shape}, expected {(n_leads, length)}"
                )
            signals[row] = self.preprocess(signal).astype(np.float16)
        signals.flush()
        del signals
        meta = {
            "version": CACHE_VERSION,
            "sampling_rate": self.sampling_rate,
            "normalize": self.normalize,
            "leads": list(LEADS),
            "length": int(length),
            "ecg_ids": self.ecg_ids,
        }
        self.meta_path.write_text(json.dumps(meta))
        return meta

    @property
    def signals(self) -> np.ndarray:
        """Memmap, opened lazily so the object pickles cheaply to workers."""
        if self._signals is None:
            self._signals = np.load(self.signals_path, mmap_mode="r")
        return self._signals

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_signals"] = None
        return state


class PTBXLDataset(Dataset):
    """Records of the given folds with multi-hot labels for ``task``.

    Args:
        root: directory holding ``ptbxl_database.csv`` and the records.
        folds: ``strat_fold`` values to include (1-10).
        task: one of :data:`TASKS`.
        sampling_rate: 100 or 500 Hz.
        leads: lead names (see :data:`LEADS`) or ``None`` for all twelve.
        normalize: ``"record"`` (per-record, per-lead z-score) or ``"none"``.
        min_likelihood: drop SCP codes below this likelihood (percent).
        drop_unlabelled: skip records with no label in ``task``.
        cache_dir: override the cache location (default ``<root>/cache/...``).

    Items are ``(signal, target)`` with ``signal`` a float32 ``(leads,
    length)`` tensor and ``target`` a float32 ``(num_classes,)`` multi-hot
    tensor.
    """

    def __init__(
        self,
        root: str | Path,
        folds: Sequence[int],
        task: str = "superdiagnostic",
        sampling_rate: int = 100,
        leads: Sequence[str] | None = None,
        normalize: str = "record",
        min_likelihood: float = 0.0,
        drop_unlabelled: bool = True,
        cache_dir: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.folds = [int(f) for f in folds]
        self.task = task
        database, statements = load_metadata(self.root)
        self.cache = PTBXLCache(
            self.root, database, sampling_rate, normalize, cache_dir
        )

        mapping = task_mapping(statements, task)
        self.classes: list[str] = sorted(set(mapping.values()))
        class_index = {name: i for i, name in enumerate(self.classes)}

        rows = database.loc[self.cache.ecg_ids]
        rows = rows[rows["strat_fold"].isin(self.folds)]
        ecg_ids: list[int] = []
        targets: list[np.ndarray] = []
        for ecg_id, codes in rows["scp_codes"].items():
            labels = aggregate_labels(codes, mapping, min_likelihood)
            if not labels and drop_unlabelled:
                continue
            target = np.zeros(len(self.classes), dtype=np.float32)
            for name in labels:
                target[class_index[name]] = 1.0
            ecg_ids.append(int(ecg_id))
            targets.append(target)
        self.ecg_ids = ecg_ids
        self.rows = np.array([self.cache.row_of[i] for i in ecg_ids], dtype=np.int64)
        self.targets = torch.from_numpy(
            np.stack(targets)
            if targets
            else np.zeros((0, len(self.classes)), np.float32)
        )

        lead_names = (
            list(LEADS) if leads is None else [str(lead).upper() for lead in leads]
        )
        unknown = [lead for lead in lead_names if lead not in LEADS]
        if unknown:
            raise ValueError(f"Unknown leads {unknown}; choose from {LEADS}")
        self.leads = lead_names
        self.lead_index = np.array([LEADS.index(lead) for lead in lead_names])

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def in_channels(self) -> int:
        return len(self.leads)

    @property
    def length(self) -> int:
        return int(self.cache.meta["length"])

    @property
    def multilabel(self) -> bool:
        return True

    def class_counts(self) -> dict[str, int]:
        counts = self.targets.sum(dim=0).long().tolist()
        return dict(zip(self.classes, counts, strict=True))

    def __len__(self) -> int:
        return len(self.ecg_ids)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        signal = self.cache.signals[self.rows[index]][self.lead_index]
        return torch.from_numpy(np.asarray(signal, dtype=np.float32)), self.targets[
            index
        ]
