from enum import Enum
from typing import Any


class BaseArgs:
    def __init__(self, data: list[(str, Any, type, bool)]):
        self.types = {}
        self.nullables = {}
        for (name, value, var_type, nullable) in data:
            setattr(self, name, value)
            self.types[name] = var_type
            self.nullables[name] = nullable

    def to_dict(self) -> dict:
        data = {}
        for (name, _) in self.types.items():
            value = getattr(self, name)
            if self.types[name] == str:
                data[name] = value
            elif issubclass(self.types[name], Enum):
                data[name] = str(value)
            elif self.types[name] == bool:
                data[name] = value
            elif self.types[name] == int:
                data[name] = value
            elif self.types[name] == float:
                if value in [float('inf'), float('-inf')]:
                    data[name] = str(value)
                else:
                    data[name] = value
            else:
                data[name] = value

        return data

    def from_dict(self, data):
        for (name, _) in self.types.items():
            try:
                if self.types[name] == str:
                    setattr(self, name, data[name])
                elif issubclass(self.types[name], Enum):
                    if isinstance(data[name], str):
                        if self.nullables[name]:
                            setattr(self, name, None if data[name] is None else self.types[name][data[name]])
                        else:
                            setattr(self, name, self.types[name][data[name]])
                    else:
                        setattr(self, name, data[name])
                elif self.types[name] == bool:
                    setattr(self, name, data[name])
                elif self.types[name] == int:
                    if self.nullables[name]:
                        setattr(self, name, None if data[name] is None else int(data[name]))
                    else:
                        setattr(self, name, int(data[name]))
                elif self.types[name] == float:
                    # check for strings to support dicts loaded from json
                    if data[name] in [float('inf'), float('-inf'), 'inf', '-inf']:
                        setattr(self, name, float(data[name]))
                    if self.nullables[name]:
                        setattr(self, name, None if data[name] is None else float(data[name]))
                    else:
                        setattr(self, name, float(data[name]))
                else:
                    setattr(self, name, data[name])
            except Exception:
                if name in data:
                    print(f"Could not set {name} as {str(data[name])}")
                else:
                    #print(f"Could not set {name}, not found.")
                    pass
        return self

    def __to_arg_name(self, var_name: str) -> str:
        return "--" + var_name.replace('_', '-')

    def __to_var_name(self, arg_name: str) -> str:
        return arg_name.lstrip('-').replace('-', '_')

    def to_args(self) -> str:
        data = []
        for (name, _) in self.types.items():
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
                else:
                    data.append(f"{self.__to_arg_name(name)}=\"{str(value)}\"")

        return ' '.join(data)
