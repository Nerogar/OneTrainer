from enum import Enum
from typing import Any

from modules.util.config.BaseConfig import BaseConfig


class BaseArgs(BaseConfig):
    def __init__(self, data: list[(str, Any, type, bool)]):
        super().__init__(data)

    def __to_arg_name(self, var_name: str) -> str:
        return "--" + var_name.replace('_', '-')

    def __to_var_name(self, arg_name: str) -> str:
        return arg_name.lstrip('-').replace('-', '_')

    def to_args(self) -> str:
        data = [self.__transform_value(name) for name in self.types]
        return ' '.join(data)

    def __transform_value(self, name: str):
        value = getattr(self, name)
        if value is None:
            return None
        if self.types[name] is str:
            return f"{self.__to_arg_name(name)}=\"{value}\""
        if issubclass(self.types[name], Enum):
            return f"{self.__to_arg_name(name)}=\"{str(value)}\""
        if self.types[name] is bool:
            if self.nullables[name]:
                return f"{self.__to_arg_name(name)}=\"{str(value)}\""
            if value:
                return self.__to_arg_name(name)
            return None
        if self.types[name] is int:
            return f"{self.__to_arg_name(name)}=\"{str(value)}\""
        if self.types[name] is float:
            if value in [float('inf'), float('-inf')]:
                return f"{self.__to_arg_name(name)}=\"{str(value)}\""
            return f"{self.__to_arg_name(name)}=\"{str(value)}\""
        if self.types[name] == list[str]:
            for val in value:
                return f"{self.__to_arg_name(name)}=\"{val}\""
            return None

        return f"{self.__to_arg_name(name)}=\"{str(value)}\""
