from typing import Union, Tuple, Optional, TypeAlias, TypeVar
from torch import nn

# TODO:
#   - check type hints for tensors as they are currently not correct?
#   - ...

Alias = TypeAlias

T = TypeVar('T')
_scalar_or_2tuple_t: Alias = Union[T, Tuple[T, T]]

stride_t: Alias = Optional[_scalar_or_2tuple_t[int]]
k_size_t: Alias = _scalar_or_2tuple_t[int]
pad_t: Alias = _scalar_or_2tuple_t[int]
dil_t: Alias = _scalar_or_2tuple_t[int]

act_t: Alias = nn.modules.activation
batch_size_t = int
channels_t = int
img_height_t = int
img_width_t = int
tensor4d_t: Alias = Tuple[batch_size_t, channels_t, img_height_t, img_width_t]
tensor_gmp_output_t: Alias = Tuple[batch_size_t, channels_t]
tensor_gap_output_t: Alias = Tuple[batch_size_t, channels_t]
