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
            arg_string = self._to_args_single(name, value)
            if arg_string is not None:
                data.append(arg_string)

        return ' '.join(data)

    def _to_args_single(self, name: str, value) -> str | None:
        if value is None:
            return None
        klass = self.types[name]
        arg_name = self.__to_arg_name(name)

        if klass is bool:
            if self.nullables[name]:
                return f"{arg_name}=\"{str(value)}\""
            if value:
                return arg_name
            return None
        if klass == list[str]:
            return ' '.join(value)

        return f"{arg_name}=\"{value!s}\""

