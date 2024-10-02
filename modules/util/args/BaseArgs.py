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
        data = []
        for name in self.types:
            value = getattr(self, name)
            if value is not None:
                if self.types[name] == str:
                    data.append(f"{self.__to_arg_name(name)}=\"{value}\"")
                elif issubclass(self.types[name], Enum):
                    data.append(f"{self.__to_arg_name(name)}=\"{str(value)}\"")
                elif self.types[name] == bool:
                    if self.nullables[name]:
                        data.append(f"{self.__to_arg_name(name)}=\"{str(value)}\"")
                    else:
                        if value:
                            data.append(self.__to_arg_name(name))
                elif self.types[name] == int:
                    data.append(f"{self.__to_arg_name(name)}=\"{str(value)}\"")
                elif self.types[name] == float:
                    if value in [float('inf'), float('-inf')]:
                        data.append(f"{self.__to_arg_name(name)}=\"{str(value)}\"")
                    else:
                        data.append(f"{self.__to_arg_name(name)}=\"{str(value)}\"")
                elif self.types[name] == list[str]:
                    for val in value:
                        data.append(f"{self.__to_arg_name(name)}=\"{val}\"")
                else:
                    data.append(f"{self.__to_arg_name(name)}=\"{str(value)}\"")

        return ' '.join(data)
