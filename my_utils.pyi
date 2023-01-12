from typing import overload, Callable, Any, Tuple

@overload
def kfold_split(k: int) -> Callable[..., Any]: ...

@overload
def kfold_split(k: int, r: Tuple[float, float, float]) -> Callable[..., Any]: ...