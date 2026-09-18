"""Download public ECG datasets from PhysioNet with ``wfdb``.

Requires the ``data`` extra: ``pip install "dnasty[data]"``.

Datasets:
    mitbih    MIT-BIH Arrhythmia Database (48 records, 2 leads, beat labels)
    ptbxl     PTB-XL v1.0.3 (21,837 12-lead 10 s records; ~1.7 GB at 100 Hz,
              ~7 GB with the 500 Hz files)
    cpsc2018  CPSC 2018 training set (6,877 12-lead records) as distributed
              in the PhysioNet/CinC 2021 challenge training data
"""

from __future__ import annotations

from pathlib import Path

PHYSIONET_DBS = {
    "mitbih": "mitdb",
    "ptbxl": "ptb-xl/1.0.3",
    "cpsc2018": "challenge-2021/1.0.3",
}
CPSC_PREFIX = "training/cpsc_2018/"


def _wfdb():
    try:
        import wfdb
    except ImportError as exc:  # pragma: no cover - depends on extras
        raise ImportError(
            "wfdb is required for downloads: pip install 'dnasty[data]'"
        ) from exc
    return wfdb


def download(name: str, data_dir: str | Path, keep_subdirs: bool = True) -> Path:
    """Download ``name`` into ``data_dir/<name>`` and return that path."""
    if name not in PHYSIONET_DBS:
        raise ValueError(f"Unknown dataset '{name}'. Choose from {list(PHYSIONET_DBS)}")
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
