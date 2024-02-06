import random
import warnings
from torch.utils.data import Subset
from typing import MutableSequence, Callable, Any
import collections.abc as abc
import functools
from datasets import CPSCDataset2D


# TODO: This is an old file that needs to be revamped


def load_2d_dataset(data_path, ref_path) -> CPSCDataset2D:
    return CPSCDataset2D(data_path, ref_path)


def split(dataset, ratio: tuple) -> tuple:
    size = len(dataset)
    indices = list(range(size))
    random.shuffle(indices)

    if len(ratio) == 3:
        train_ratio, eval_ratio, _ = ratio
        train_size, eval_size = int(size * train_ratio), int(size * eval_ratio)
        train_set = Subset(dataset, indices[:train_size])
        eval_set = Subset(dataset, indices[train_size:train_size + eval_size])
        test_set = Subset(dataset, indices[train_size + eval_size:])
        return train_set, eval_set, test_set
    else:
        train_size = int(size * ratio[0])
        train_set = Subset(dataset, indices[:train_size])
        eval_set = Subset(dataset, indices[train_size:])
        return train_set, eval_set


def split_dataset(dataset: CPSCDataset2D,
                  ksplit: tuple[int, tuple]) -> tuple[tuple]:
    if ksplit[0] > 1:
        return tuple(split(dataset, ksplit[1]) for _ in range(ksplit[0]))
    return split(dataset, ksplit[1])


# decorator to apply kfold split on a load_dataset() function
def ksplit(k: int, ratio: abc.Sequence) -> Callable[..., Any]:
    if ratio[0] < ratio[1] or ratio[0] < ratio[2]:
        message = f"splits may not be correct: (train, eval, test) = {ratio}"
        warnings.warn(message)

    if not isinstance(ratio, MutableSequence):
        ratio = list(ratio)

    epsilon = 1e-3
    if (sum(ratio) - 1) > epsilon:
        ratio = [round(x / sum(ratio), 2) for x in ratio]

    if k < 0:
        warnings.warn(
            f"You supplied an invalid value for k ({k})... resorting to "
            f"default k = 1")
        k = 1

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def inner(*args, **kwargs) -> tuple:
            dataset = func(*args, **kwargs)
            return split_dataset(dataset, ratio, k)

        return inner

    return decorator
