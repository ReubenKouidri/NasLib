from __future__ import annotations

import json
from collections import abc
from pathlib import Path
from typing import Any

import yaml


def _convert_type(value: Any) -> Any:
    """Convert a string to int, float, bool or None where possible.

    Non-string values (already typed, e.g. from YAML) are returned unchanged.
    """
    if not isinstance(value, str):
        return value
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    lowered = value.lower()
    if lowered == "none":
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    return value


class Config:
    """Attribute-style access to a nested mapping loaded from YAML or JSON.

    String leaves are converted to int/float/bool/None where they parse as
    such, so legacy JSON configs with quoted numbers keep working.

    Example::

        >>> cfg = Config({"train": {"epochs": "10"}, "seed": 0})
        >>> cfg.train.epochs, cfg.seed
        (10, 0)
    """

    def __init__(self, arg: abc.Mapping) -> None:
        if not isinstance(arg, abc.Mapping):
            raise TypeError(f"Config must be a mapping, not {type(arg).__name__}.")
        self.__dict__.update({k: self._process_entry(v) for k, v in arg.items()})

    @staticmethod
    def _process_entry(entry: Any) -> Any:
        if isinstance(entry, abc.Mapping):
            return Config(entry)
        if isinstance(entry, list):
            return [Config._process_entry(item) for item in entry]
        return _convert_type(entry)

    @classmethod
    def from_file(cls, file_path: str | Path) -> Config:
        path = Path(file_path)
        with path.open() as f:
            if path.suffix in {".yaml", ".yml"}:
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        return cls(data or {})

    def to_dict(self) -> dict[str, Any]:
        def unwrap(value: Any) -> Any:
            if isinstance(value, Config):
                return value.to_dict()
            if isinstance(value, list):
                return [unwrap(v) for v in value]
            return value

        return {k: unwrap(v) for k, v in self.__dict__.items()}

    def get(self, item: str, default: Any = None) -> Any:
        return self.__dict__.get(item, default)

    def __contains__(self, item: str) -> bool:
        return item in self.__dict__

    def __getattr__(self, item: str) -> Any:
        if item in self.__dict__:
            return self.__dict__[item]
        raise AttributeError(f"'Config' object has no attribute '{item}'")

    def __repr__(self) -> str:
        items = []
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                items.append(f"\n  {key}={value!r}")
            else:
                items.append(f"{key}: {value!r}")
        return f"Config({', '.join(items)})"
