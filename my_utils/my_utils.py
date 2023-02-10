import numpy as np
from typing import Callable, Any, overload, Tuple, List
import torch
from torch.utils.data import Subset
import os
import functools
from my_datasets.CPSCDataset import CPSCDataset2D


def get_num_correct(preds, tgts):
    return preds.argmax(dim=1).eq(tgts).sum().item()


@overload
def split(k: int) -> Callable[..., Any]: ...


def split(k: int = 10) -> Callable[..., Any]:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        if isinstance(k, int):
            if k < 1:
                raise ValueError("k must be larger than 1 in k-fold split")

        @functools.wraps(func)
        def inner(*args, **kwargs) -> List[Tuple[Subset, ...]]:
            splits = []
            dataset = func(*args, **kwargs)
            chunk_size = len(dataset) // k
            dataset_length = len(dataset)
            for i in range(k):
                start_idx = i * chunk_size
                if i < k - 1:
                    end_idx = (i + 1) * chunk_size
                    trainset = Subset(
                        dataset, indices=list(
                            list(range((end_idx + chunk_size), dataset_length)) +
                            list(range(0, start_idx))
                            )
                        )
                    valset = Subset(dataset, indices=range(end_idx, (end_idx + chunk_size)))
                    testset = Subset(dataset, indices=range(start_idx, end_idx))
                    splits.append((trainset, valset, testset))
                else:
                    end_idx = 0
                    testset = Subset(dataset, indices=range(start_idx, len(dataset)))
                    valset = Subset(dataset, indices=range(end_idx, (end_idx + chunk_size)))
                    trainset = Subset(dataset, indices=list(range((end_idx + chunk_size), start_idx)))
                    splits.append((trainset, valset, testset))

            return splits
        return inner
    return decorator


def load_dataset(fp, rp) -> CPSCDataset2D:
    if not os.path.exists(fp):
        fp = os.path.join(os.getcwd(), fp)
    if not os.path.exists(rp):
        rp = os.path.join(os.getcwd(), rp)

    return CPSCDataset2D(fp, rp)


class ModelSave:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.path = path
        self.delta = delta
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            if self.verbose:
                self.trace_func("No decrease in val loss...")
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...'
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class Trainer:
    def __init__(self) -> None:
        self.checkpoint = ModelSave()
        self.stats = {}

    def train(self, model: torch.utils.data.dataset): ...

