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
            if isinstance(value, str):
                data[name] = value
            elif isinstance(value, Enum):
                data[name] = str(value)
            elif isinstance(value, bool):
                data[name] = value
            elif isinstance(value, int):
                data[name] = value
            elif isinstance(value, float):
                data[name] = value
            elif isinstance(value, tuple) and len(value) == 2:
                data[name] = {"x": value[0], "y": value[1]}
            elif value is None:
                data[name] = "None"
            elif value in [float('inf'), float('-inf'), int('inf'), int('-inf')]:
                data[name] = str(value)
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
                    if self.nullables[name]:
                        setattr(self, name, None if data[name] is None else float(data[name]))
                    else:
                        setattr(self, name, float(data[name]))
                elif isinstance(data[name], tuple) and "x" in data[name] and "y" in data[name]:
                    setattr(self, name, (data[name]["x"], data[name]["y"]))
                elif data[name] == "None":
                    setattr(self, name, data[name])
                elif data[name] in ["inf", "-inf"]:
                    if isinstance(data[name], int):
                        setattr(self, name, int(float(data[name])))
                    else:
                        setattr(self, name, float(data[name]))
                else:
                    setattr(self, name, data[name])
            except Exception:
                if name in data:
                    print(f"Could not set {name} as {str(data[name])}")
                else:
                    print(f"Could not set {name}, not found.")
        return self

    def __to_arg_name(self, var_name: str) -> str:
        return "--" + var_name.replace('_', '-')

    def __to_var_name(self, arg_name: str) -> str:
        return arg_name.lstrip('-').replace('-', '_')

    def to_args(self) -> str:
        data = []
        for (name, _) in self.types.items():
            value = getattr(self, name)
            if isinstance(value, str):
                data.append(f"{self.__to_arg_name(name)}=\"{value}\"")
            elif isinstance(value, Enum):
                data.append(f"{self.__to_arg_name(name)}=\"{str(value)}\"")
            elif isinstance(value, bool):
                if value:
                    data.append(self.__to_arg_name(name))
            elif isinstance(value, int):
                data.append(f"{self.__to_arg_name(name)}=\"{str(value)}\"")
            elif isinstance(value, float):
                data.append(f"{self.__to_arg_name(name)}=\"{str(value)}\"")
            elif value is not None:
                data.append(f"{self.__to_arg_name(name)}=\"{str(value)}\"")
            elif isinstance(value, tuple) and len(value) == 2:
                data.append(f"{self.__to_arg_name(name)}=\"({value[0]}, {value[1]})\"")
            elif value is None:
                data.append(f"{self.__to_arg_name(name)}=\"None\"")
            elif value in [float('inf'), float('-inf'), int('inf'), int('-inf')]:
                data.append(f"{self.__to_arg_name(name)}=\"{str(value)}\"")
            elif value is not None:
                data.append(f"{self.__to_arg_name(name)}=\"{str(value)}\"")

        return ' '.join(data)
