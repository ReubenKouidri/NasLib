import random
import warnings
from torch.utils.data import Subset, Dataset
from typing import MutableSequence, TypeVar, Callable, Any
import collections.abc as abc
import functools
from my_datasets.CPSCDataset import CPSCDataset2D
random.seed(9834275)

T_co = TypeVar('T_co', covariant=True)


def load_2d_dataset(data_path, ref_path) -> CPSCDataset2D:
    return CPSCDataset2D(data_path, ref_path)


def load_datasets(data_path, ref_path):
    return load_2d_dataset(data_path, ref_path)


def split(dataset, ratio) -> tuple:
    size = len(dataset)
    indices = list(range(size))
    random.shuffle(indices)
    split_sizes = (int(size * ratio[0]), int(size * ratio[1]), int(size * ratio[2]))

    train_indices = indices[0:split_sizes[0]]
    eval_indices = indices[split_sizes[0]:split_sizes[0] + split_sizes[1]]
    test_indices = indices[split_sizes[0] + split_sizes[1]:]

    train_set = Subset(dataset, indices=list(train_indices))
    eval_set = Subset(dataset, indices=list(eval_indices))
    test_set = Subset(dataset, indices=list(test_indices))

    return train_set, eval_set, test_set


def split_dataset(dataset: abc.Sequence, n: int, ratio: abc.Sequence) -> tuple[tuple]:
    return tuple(split(dataset, ratio) for _ in range(n))


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
        warnings.warn(f"You supplied an invalid value for k ({k})... resorting to default k = 1")
        k = 1

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def inner(*args, **kwargs) -> tuple:
            dataset = func(*args, **kwargs)
            return split_dataset(dataset, ratio, k)
        return inner
    return decorator
