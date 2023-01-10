from torch import nn
from torch import Tensor
from typing import Union, Tuple, Optional, TypeVar, Callable


# TODO:
#  - check type hints for tensors as they are currently not correct?
#  - ...

T = TypeVar('T')
_scalar_or_2tuple_t = Union[T, Tuple[T, T]]

stride_t = Optional[_scalar_or_2tuple_t[int]]
k_size_t = _scalar_or_2tuple_t[int]
pad_t = _scalar_or_2tuple_t[int]
dil_t = _scalar_or_2tuple_t[int]

act_t = Callable[..., Tensor]
batch_size_t = int
channels_t = int
img_height_t = int
img_width_t = int
tensor4d_t = Tuple[batch_size_t, channels_t, img_height_t, img_width_t]
tensor_gmp_output_t = Tuple[batch_size_t, channels_t]
tensor_gap_output_t = Tuple[batch_size_t, channels_t]
