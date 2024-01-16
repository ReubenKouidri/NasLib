from __future__ import annotations
from typing import Any
from collections import abc
import json


def _convert_type(value: str) -> Any:
    """Convert string to int, float, bool, NoneType, or leave as str."""
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


class Config:
    """
    A configuration handler class for loading and accessing JSON
    configuration data.

    This class converts JSON data into a Python object, allowing
    for attribute-style access. It supports nested configurations
    and automatically converts string values to appropriate Python
    data types (int, float, bool, None, or str).

    The class can be initialized with a dictionary representing
    configuration data, typically loaded from a JSON file.

    Attributes:
        __dict__ (dict): A dictionary holding configuration
                         keys and values.

    Methods:
        __init__(arg): Initialize the Config instance with a mapping object.
        _create_config(entry): Class method to create Config objects from
        JSON entries.
        from_file(file_path): Class method to load configuration from a
        JSON file.
        __getattr__(item): Special method to allow attribute-style access.
        __repr__(): Special method to provide a string representation of
        the object.
    """

    def __init__(self, arg):
        if isinstance(arg, abc.Mapping):
            self.__dict__.update(
                {k: self._process_entry(v) for k, v in arg.items()})
        else:
            raise TypeError(
                f"Config must be a mapping, not {type(arg).__name__}.")

    @staticmethod
    def _process_entry(entry):
        if isinstance(entry, abc.MutableMapping):
            return Config(entry)
        elif isinstance(entry, list):
            return [Config._process_entry(item) if
                    isinstance(item, abc.MutableMapping) else
                    _convert_type(item) for item in entry]
        else:
            return _convert_type(entry)

    @classmethod
    def from_file(cls, file_path: str) -> Config:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return cls(data)

    def __getattr__(self, item):
        if item in self.__dict__:
            return self.__dict__[item]
        raise AttributeError(f"'Config' object has no attribute '{item}'")

    def __repr__(self):
        return f"Config({self.__dict__!r})"
