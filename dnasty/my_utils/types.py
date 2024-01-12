from typing import Optional, TypeVar, Union, Tuple

__all__ = [
    'size_1_t',
    'size_2_t',
    'size_2_opt_t',
    'size_any_t',
    'size_any_opt_t'
]

T = TypeVar('T')
_scalar_or_tuple_any_t = Union[T, Tuple[T, ...]]
_scalar_or_tuple_1_t = Union[T, Tuple[T]]
_scalar_or_tuple_2_t = Union[T, Tuple[T, T]]

# For arguments which represent size parameters (eg, kernel size, padding)
size_any_t = _scalar_or_tuple_any_t[int]
size_1_t = _scalar_or_tuple_1_t[int]
size_2_t = _scalar_or_tuple_2_t[int]

# For arguments which represent optional size parameters
# (eg, adaptive pool parameters, padding with default value, etc.)
size_any_opt_t = _scalar_or_tuple_any_t[Optional[int]]
size_2_opt_t = _scalar_or_tuple_2_t[Optional[int]]
