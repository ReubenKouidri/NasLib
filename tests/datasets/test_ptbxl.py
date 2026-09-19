"""PTB-XL loader tests on a synthetic directory with the PhysioNet layout."""

import json

import numpy as np
import pandas as pd
import pytest
import torch

from dnasty.data.ptbxl import (
    LEADS,
    PTBXLDataset,
    aggregate_labels,
    load_metadata,
    task_mapping,
)

wfdb = pytest.importorskip("wfdb")

FS = 100
LENGTH = 200  # 2 s instead of 10 s keeps the fixture small


def _write_record(root, ecg_id, rng):
    name = f"records100/00000/{ecg_id:05d}_lr"
    (root / "records100" / "00000").mkdir(parents=True, exist_ok=True)
    # lead-dependent scale so that per-lead standardisation is observable
    signal = rng.normal(size=(LENGTH, 12)) * np.arange(1, 13) + 5.0
    wfdb.wrsamp(
        f"{ecg_id:05d}_lr",
        fs=FS,
        units=["mV"] * 12,
        sig_name=list(LEADS),
        p_signal=signal,
        fmt=["16"] * 12,
        write_dir=str(root / "records100" / "00000"),
    )
    return name


@pytest.fixture(scope="module")
def ptbxl_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("ptbxl")
    rng = np.random.default_rng(0)
    statements = pd.DataFrame(
        {
            "description": [
                "normal",
                "anterior MI",
                "inferior MI",
                "LBBB",
                "sinus",
                "low QRS",
            ],
            "diagnostic": [1.0, 1.0, 1.0, 1.0, np.nan, np.nan],
            "form": [np.nan, np.nan, np.nan, np.nan, np.nan, 1.0],
            "rhythm": [np.nan, np.nan, np.nan, np.nan, 1.0, np.nan],
            "diagnostic_class": ["NORM", "MI", "MI", "CD", np.nan, np.nan],
            "diagnostic_subclass": ["NORM", "AMI", "IMI", "LBBB", np.nan, np.nan],
        },
        index=pd.Index(["NORM", "AMI", "IMI", "LBBB", "SR", "LVOLT"]),
    )
    statements.to_csv(root / "scp_statements.csv")

    records = []
    scp = [
        {"NORM": 100.0, "SR": 0.0},
        {"AMI": 80.0},
        {"IMI": 50.0, "LBBB": 100.0},
        {"SR": 0.0},  # rhythm only: no superdiagnostic label
        {"NORM": 15.0},
        {"LBBB": 100.0, "LVOLT": 0.0},
        {"AMI": 100.0},
        {"NORM": 100.0},
    ]
    folds = [1, 1, 2, 2, 9, 9, 10, 10]
    for i, (codes, fold) in enumerate(zip(scp, folds, strict=True), start=1):
        filename = (
            _write_record(root, i, rng) if i != 8 else f"records100/00000/{i:05d}_lr"
        )
        records.append(
            {
                "ecg_id": i,
                "patient_id": 1000 + i,
                "scp_codes": str(codes),
                "strat_fold": fold,
                "filename_lr": filename,
                "filename_hr": filename.replace("records100", "records500").replace(
                    "_lr", "_hr"
                ),
            }
        )
    pd.DataFrame(records).set_index("ecg_id").to_csv(root / "ptbxl_database.csv")
    return root


def test_task_mappings(ptbxl_root):
    _, statements = load_metadata(ptbxl_root)
    assert task_mapping(statements, "superdiagnostic") == {
        "NORM": "NORM",
        "AMI": "MI",
        "IMI": "MI",
        "LBBB": "CD",
    }
    assert set(task_mapping(statements, "subdiagnostic").values()) == {
        "NORM",
        "AMI",
        "IMI",
        "LBBB",
    }
    assert set(task_mapping(statements, "rhythm")) == {"SR"}
    assert set(task_mapping(statements, "form")) == {"LVOLT"}
    assert len(task_mapping(statements, "all")) == 6
    with pytest.raises(ValueError):
        task_mapping(statements, "nope")


def test_aggregate_labels_respects_likelihood():
    mapping = {"NORM": "NORM", "AMI": "MI"}
    codes = {"NORM": 15.0, "AMI": 0.0, "SR": 0.0}
    assert aggregate_labels(codes, mapping) == {"NORM", "MI"}
    assert aggregate_labels(codes, mapping, min_likelihood=10) == {"NORM"}


def test_dataset_folds_labels_and_shapes(ptbxl_root):
    train = PTBXLDataset(ptbxl_root, folds=[1, 2])
    assert train.classes == ["CD", "MI", "NORM"]
    # record 4 has only a rhythm code and is dropped
    assert train.ecg_ids == [1, 2, 3]
    assert train.targets.tolist() == [[0, 0, 1], [0, 1, 0], [1, 1, 0]]
    x, y = train[0]
    assert x.shape == (12, LENGTH) and x.dtype == torch.float32
    assert y.shape == (3,) and y.dtype == torch.float32
    assert train.num_classes == 3 and train.in_channels == 12 and train.length == LENGTH
    assert train.multilabel
    assert train.class_counts() == {"CD": 1, "MI": 2, "NORM": 1}

    keep_unlabelled = PTBXLDataset(ptbxl_root, folds=[1, 2], drop_unlabelled=False)
    assert keep_unlabelled.ecg_ids == [1, 2, 3, 4]


def test_missing_records_are_skipped(ptbxl_root):
    # record 8 is listed in the CSV but has no files on disk
    test = PTBXLDataset(ptbxl_root, folds=[10])
    assert test.ecg_ids == [7]


def test_standardisation_and_lead_selection(ptbxl_root):
    ds = PTBXLDataset(ptbxl_root, folds=[1])
    x, _ = ds[0]
    assert torch.allclose(x.mean(dim=1), torch.zeros(12), atol=2e-2)
    assert torch.allclose(x.std(dim=1), torch.ones(12), atol=5e-2)

    raw = PTBXLDataset(ptbxl_root, folds=[1], normalize="none")
    x_raw, _ = raw[0]
    assert x_raw.mean().item() > 3.0  # offset of 5 mV survives

    subset = PTBXLDataset(ptbxl_root, folds=[1], leads=["II", "v5"])
    x_sub, _ = subset[0]
    assert subset.in_channels == 2
    assert torch.equal(x_sub[0], x[1]) and torch.equal(x_sub[1], x[10])
    with pytest.raises(ValueError):
        PTBXLDataset(ptbxl_root, folds=[1], leads=["XX"])


def test_cache_is_reused_without_wfdb(ptbxl_root, monkeypatch):
    first = PTBXLDataset(ptbxl_root, folds=[1])
    meta = json.loads(first.cache.meta_path.read_text())
    assert meta["ecg_ids"] == [1, 2, 3, 4, 5, 6, 7]

    def boom(*args, **kwargs):
        raise AssertionError("cache should have been reused")

    monkeypatch.setattr(wfdb, "rdsamp", boom)
    second = PTBXLDataset(ptbxl_root, folds=[9])
    assert second.cache.dir == first.cache.dir
    assert second.ecg_ids == [5, 6]
    row = first.cache.row_of[5]
    expected = torch.from_numpy(np.asarray(first.cache.signals[row], dtype=np.float32))
    assert torch.equal(second[0][0], expected)


def test_dataset_pickles_without_the_memmap(ptbxl_root):
    import pickle

    ds = PTBXLDataset(ptbxl_root, folds=[1])
    _ = ds[0]  # opens the memmap
    clone = pickle.loads(pickle.dumps(ds))
    assert clone.cache._signals is None
    assert torch.equal(clone[0][0], ds[0][0])


def test_no_records_raises(tmp_path):
    (tmp_path / "scp_statements.csv").write_text("code,diagnostic\nNORM,1.0\n")
    pd.DataFrame(
        {
            "ecg_id": [1],
            "scp_codes": ["{'NORM': 100.0}"],
            "strat_fold": [1],
            "filename_lr": ["records100/00000/00001_lr"],
            "filename_hr": ["records500/00000/00001_hr"],
        }
    ).set_index("ecg_id").to_csv(tmp_path / "ptbxl_database.csv")
    with pytest.raises(FileNotFoundError, match="dnasty data download"):
        PTBXLDataset(tmp_path, folds=[1])
