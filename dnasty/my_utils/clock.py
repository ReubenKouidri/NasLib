from typing import Callable, Any
import functools
import time


def clock(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Better version that:
        - does not mask __name__ and __doc__ of the decorated function
        - takes **kwargs
    """

    @functools.wraps(func)
    def clocked(*args, **kwargs):
        # print(f"ARGS: {args}")
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        name = func.__name__
        arg_list = []
        if args:
            arg_list.append(', '.join(
                repr(arg) for arg in args))  # joins ', ' to the end of 2nd part
        if kwargs:
            pairs = [f"{k}={w}" for k, w in sorted(kwargs.items())]
            arg_list.append(', '.join(pairs))
        arg_str = ', '.join(arg_list)
        print(f"[{elapsed:.8f}s] {name}({arg_str}) -> {result}")
        return result

    return clocked
