import numpy as np
import pytest
import torch

from dnasty.data import CPSCDataset, CPSCDataset2D, read_reference
from dnasty.utils.wavelets import mexh

pytestmark = pytest.mark.data


@pytest.fixture(scope="module")
def dataset_1d(cpsc_paths):
    data_dir, reference = cpsc_paths
    return CPSCDataset(data_dir=data_dir, reference_path=reference, lead=3)


@pytest.fixture(scope="module")
def dataset_2d(cpsc_paths):
    data_dir, reference = cpsc_paths
    return CPSCDataset2D(
        data_dir=data_dir, reference_path=reference, wavelet="mexh", lead=3
    )


def test_reference_join(cpsc_paths):
    data_dir, reference = cpsc_paths
    ref = read_reference(reference)
    assert ref["A0001"] == (4, -1, -1)  # PAC=5 in the CSV -> class 4, no extras
    ds = CPSCDataset(data_dir=data_dir, reference_path=reference)
    assert ds.record_ids[0] == "A0001"
    assert int(ds[0][1]) == 4


def test_1d_shapes_and_types(dataset_1d):
    x, y = dataset_1d[0]
    assert torch.is_tensor(x) and torch.is_tensor(y)
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64
    assert x.shape == torch.Size([1000])  # 4 s at 500 Hz, decimated by 2
    assert y.shape == torch.Size([])
    assert len(dataset_1d) == 100


def test_test_mode_returns_all_labels(dataset_1d):
    dataset_1d.test = True
    try:
        _, y = dataset_1d[0]
        assert y.shape == torch.Size([3])
        assert y[0] >= 0 and y[1] == -1
    finally:
        dataset_1d.test = False


def test_2d_shapes(dataset_2d):
    x, y = dataset_2d[0]
    assert x.dtype == torch.float32
    assert x.shape == torch.Size([1, 128, 128])
    assert y.shape == torch.Size([])
    assert torch.isfinite(x).all()


def test_wavelet():
    img = mexh(np.random.default_rng(0).random(1000), 64)
    assert img.shape == (128, 128)
    assert np.all(np.isfinite(img))
