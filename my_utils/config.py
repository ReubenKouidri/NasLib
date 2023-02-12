from collections import abc
from keyword import iskeyword
from typing import Union, Any, List, Type, TypeVar


TConfig = TypeVar("TConfig")


class ConfigObj:
    def __init__(self, value):
        self.value = value


class Config:
    """ object providing read-only access to configurations """

    def __new__(cls, arg: Union[abc.MutableSequence, Any]) -> Union[TConfig, List[TConfig], Any]:
        if isinstance(arg, abc.Mapping):
            return super().__new__(cls)
        elif isinstance(arg, abc.MutableSequence):
            return [cls(item) for item in arg]
        else:
            return arg

    def convert_to_int(self, d: dict) -> dict:
        for key, value in d.items():
            if isinstance(value, dict):
                d[key] = self.convert_to_int(value)
            elif isinstance(value, str):
                try:
                    d[key] = int(value)
                except ValueError:
                    try:
                        d[key] = float(value)
                    except ValueError:
                        pass
        return d

    def __init__(self, mapping) -> None:
        self.__data = self.convert_to_int(dict(mapping))

    def __getattr__(self, name):
        if hasattr(self.__data, name):
            return getattr(self.__data, name)
        else:
            return Config(self.__data[name])

    '''def __getattr__(self, name: str) -> Union[TConfig, Any]:
        """
        This method is called when an attribute is accessed on an object,
        but it is not found in the object's state dictionary
        Checks the __data dictionary for the attribute and returns the attribute using getattr if present
        Else builds a new Config object
        """
        if hasattr(self.__data, name):
            return getattr(self.__data, name)
        else:
            return Config.build(self.__data[name])'''

    '''@classmethod
    def build(cls, obj: Any) -> Union[TConfig, List[TConfig], Any]:
        """
        - If obj instance is a mapping then return a Config obj by passing it directly to constructor
        - This is because an abc.Mapping is, or can be converted to a dict directly
        - If Mutable sequence, it must be a list (only collection types is a json file are dict and list)
            => return a list of Config objects
        - If neither of these, return obj as is
        """
        if isinstance(obj, abc.Mapping):
            return cls(obj)
        elif isinstance(obj, abc.MutableSequence):
            return [cls.build(item) for item in obj]
        else:
            return obj'''
