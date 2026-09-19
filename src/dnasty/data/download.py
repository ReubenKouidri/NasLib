"""Download public ECG datasets from PhysioNet.

Reading the records needs the ``data`` extra: ``pip install "dnasty[data]"``.

Datasets:
    mitbih    MIT-BIH Arrhythmia Database (48 records, 2 leads, beat labels)
    ptbxl     PTB-XL v1.0.3 (21,837 12-lead 10 s records; ~1.7 GB at 100 Hz,
              ~7 GB with the 500 Hz files). Supports subsets by fold and by
              number of records per fold, about 25 kB per record at 100 Hz.
    cpsc2018  CPSC 2018 training set (6,877 12-lead records) as distributed
              in the PhysioNet/CinC 2021 challenge training data

``wfdb.dl_database`` resolves the latest version of a database itself, so
the names below carry no version suffix.
"""

from __future__ import annotations

import logging
import urllib.request
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

logger = logging.getLogger(__name__)

PHYSIONET_DBS = {
    "mitbih": "mitdb",
    "ptbxl": "ptb-xl",
    "cpsc2018": "challenge-2021",
}
CPSC_PREFIX = "training/cpsc_2018/"
PTBXL_VERSION = "1.0.3"
PTBXL_FILES_URL = f"https://physionet.org/files/ptb-xl/{PTBXL_VERSION}/"
PTBXL_METADATA = ("ptbxl_database.csv", "scp_statements.csv")


def _wfdb():
    try:
        import wfdb
    except ImportError as exc:  # pragma: no cover - depends on extras
        raise ImportError(
            "wfdb is required for downloads: pip install 'dnasty[data]'"
        ) from exc
    return wfdb


def _fetch(url: str, target: Path, overwrite: bool = False) -> bool:
    """Download ``url`` to ``target``; returns whether a transfer happened."""
    if target.is_file() and target.stat().st_size > 0 and not overwrite:
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as response, tmp.open("wb") as f:
        while chunk := response.read(1 << 16):
            f.write(chunk)
    tmp.replace(target)
    return True


def fetch_files(
    base_url: str, root: Path, relative_paths: Iterable[str], workers: int = 8
) -> int:
    """Fetch ``relative_paths`` under ``base_url`` into ``root`` in parallel."""
    from tqdm import tqdm

    paths = list(relative_paths)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(
            tqdm(
                pool.map(lambda p: _fetch(base_url + p, root / p), paths),
                total=len(paths),
                desc="download",
                unit="file",
            )
        )
    return sum(results)


def download_ptbxl(
    data_dir: str | Path,
    sampling_rate: int = 100,
    folds: Sequence[int] | None = None,
    limit_per_fold: int | None = None,
    workers: int = 8,
) -> Path:
    """Download PTB-XL metadata and (a subset of) the records.

    Args:
        data_dir: the records go to ``data_dir/ptbxl``.
        sampling_rate: 100 or 500 Hz files.
        folds: ``strat_fold`` values to fetch; default all ten.
        limit_per_fold: at most this many records per fold, taken in
            ``ecg_id`` order so that repeated calls extend the same subset.
        workers: parallel connections.
    """
    import pandas as pd

    if sampling_rate not in (100, 500):
        raise ValueError("sampling_rate must be 100 or 500")
    target = Path(data_dir) / "ptbxl"
    fetch_files(PTBXL_FILES_URL, target, PTBXL_METADATA, workers=2)
    database = pd.read_csv(target / "ptbxl_database.csv", index_col="ecg_id")
    if folds is not None:
        database = database[database["strat_fold"].isin([int(f) for f in folds])]
    if limit_per_fold is not None:
        database = database.sort_index().groupby("strat_fold").head(limit_per_fold)
    column = "filename_lr" if sampling_rate == 100 else "filename_hr"
    files = [f"{name}{ext}" for name in database[column] for ext in (".hea", ".dat")]
    logger.info("Fetching %d PTB-XL records into %s", len(database), target)
    fetched = fetch_files(PTBXL_FILES_URL, target, files, workers=workers)
    logger.info("%d files fetched, %d already present", fetched, len(files) - fetched)
    return target


def download(
    name: str, data_dir: str | Path, keep_subdirs: bool = True, **kwargs
) -> Path:
    """Download ``name`` into ``data_dir/<name>`` and return that path.

    ``ptbxl`` accepts ``sampling_rate``, ``folds`` and ``limit_per_fold``
    (see :func:`download_ptbxl`); the others fetch the whole database.
    """
    if name not in PHYSIONET_DBS:
        raise ValueError(f"Unknown dataset '{name}'. Choose from {list(PHYSIONET_DBS)}")
    if name == "ptbxl":
        return download_ptbxl(data_dir, **kwargs)
    if kwargs:
        raise ValueError(f"{name} does not accept options {sorted(kwargs)}")

    wfdb = _wfdb()
    target = Path(data_dir) / name
    target.mkdir(parents=True, exist_ok=True)
    db = PHYSIONET_DBS[name]

    if name == "cpsc2018":
        # The challenge RECORDS file lists directories; each holds its own RECORDS.
        groups = [g for g in wfdb.get_record_list(db) if g.startswith(CPSC_PREFIX)]
        for group in groups:
            wfdb.dl_database(f"{db}/{group}", str(target / group), keep_subdirs=False)
    else:
        wfdb.dl_database(db, str(target), keep_subdirs=keep_subdirs)
    return target
