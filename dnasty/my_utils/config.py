from __future__ import annotations
from typing import Any
from collections import abc
import json


def _load_json(json_file: str) -> dict:
    with open(json_file) as file:
        return json.load(file)


class Config:
    """ object providing read-only access to configurations """

    @classmethod
    def from_file(cls, string: str) -> Config:
        return cls(_load_json(string))

    def __new__(cls,
                arg: abc.MutableSequence | Any) -> Config | list[Config] | Any:
        if isinstance(arg, abc.Mapping):
            return super().__new__(cls)
        elif isinstance(arg, abc.MutableSequence):
            return [cls(item) for item in arg]
        else:
            return arg

    def __init__(self, mapping) -> None:
        if isinstance(mapping, Config):
            self.__data = mapping.__data
        elif isinstance(mapping, abc.Mapping):
            self.__data = self._convert(dict(mapping))
        else:
            raise TypeError(
                f"Config must be a mapping or sequence, not "
                f"{type(mapping).__name__}."
            )

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
    def _convert_type(value: str) -> Any:
        """Try to convert to int, float, bool, NoneType, or leave as str."""
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                if value.lower() == "none":
                    return None
                if value.lower() == "true":
                    return True
                if value.lower() == "false":
                    return False
        return value
