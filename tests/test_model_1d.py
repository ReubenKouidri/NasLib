import torch

from dnasty.defaults import ECGNet1d


def test_ecgnet1d_shapes_and_params():
    model = ECGNet1d(in_channels=12, num_classes=5)
    out = model(torch.randn(4, 12, 1000))
    assert out.shape == (4, 5)
    # same padding + global average pooling: any input length works
    assert model(torch.randn(2, 12, 333)).shape == (2, 5)
    n_params = sum(p.numel() for p in model.parameters())
    assert 50_000 < n_params < 200_000
