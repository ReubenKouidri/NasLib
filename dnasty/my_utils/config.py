from __future__ import annotations

from collections import abc
from typing import Any


class Config:
    """ object providing read-only access to configurations """
    def __new__(cls, arg: abc.MutableSequence | Any) -> 'Config' | list['Config'] | Any:
        if isinstance(arg, abc.Mapping):
            return super().__new__(cls)
        elif isinstance(arg, abc.MutableSequence):
            return [cls(item) for item in arg]
        else:
            return arg

    def __init__(self, mapping) -> None:
        self.__data = self._convert(dict(mapping))

    # def __getattr__(self, name):
    #     if hasattr(self.__data, name):
    #         return getattr(self.__data, name)
    #     else:
    #         return Config(self.__data[name])

    def __getattr__(self, name):
        if name in self.__data:
            return Config(self.__data[name])
        else:
            raise AttributeError(f"'Config' object has no attribute '{name}'")

    def _convert(self, d: dict) -> dict:
        for key, value in d.items():
            if isinstance(value, dict):
                d[key] = self._convert(value)
            elif isinstance(value, str):
                d[key] = self._convert_type(value)
        return d

    @staticmethod
    def _convert_type(value: str):
        """ Try to convert to int; if fails, try to convert to float; if
        fails, try NoneType; else leave as str """
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                if value == "None":
                    value = None
                else:
                    pass
        return value
