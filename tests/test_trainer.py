import torch
from torch import nn
from torch.utils.data import TensorDataset

from dnasty.data import DataModule
from dnasty.defaults import Trainer
from dnasty.utils import Config


def _config():
    return Config({"train": {"lr": 0.01, "batch_size": 4}, "eval": {"batch_size": 8}})


def _dataset(n=40):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(n, 3, generator=g)
    y = (x[:, 0] > 0).long()  # linearly separable
    return TensorDataset(x, y)


def test_split_is_seeded_and_shared():
    ds = _dataset()
    a = DataModule(ds, seed=1, train_batch_size=4, eval_batch_size=8)
    b = DataModule(ds, seed=1, train_batch_size=4, eval_batch_size=8)
    c = DataModule(ds, seed=2, train_batch_size=4, eval_batch_size=8)
    assert a.valset.indices == b.valset.indices
    assert a.valset.indices != c.valset.indices
    assert len(a.trainset) + len(a.valset) == len(ds)


def test_accuracy_counts_samples_not_batches():
    """A perfect classifier must score exactly 1.0 with batch_size > 1."""
    ds = _dataset()
    dm = DataModule(ds, seed=0, train_batch_size=4, eval_batch_size=8)
    trainer = Trainer(_config(), dm)

    class Oracle(nn.Module):
        def forward(self, x):
            return torch.stack([-x[:, 0], x[:, 0]], dim=1) * 100

    _, acc = trainer._eval(Oracle(), nn.CrossEntropyLoss())
    assert acc == 1.0


def test_fit_learns_and_returns_best_accuracy():
    ds = _dataset(200)
    dm = DataModule(ds, seed=0, train_batch_size=8, eval_batch_size=16)
    trainer = Trainer(_config(), dm)
    model = nn.Linear(3, 2)
    result = trainer.fit_detailed(model, epochs=5)
    assert 0.0 <= result.best_val_acc <= 1.0
    assert result.best_val_acc == max(e.val_acc for e in result.epochs)
    assert result.epochs[-1].train_loss < result.epochs[0].train_loss
