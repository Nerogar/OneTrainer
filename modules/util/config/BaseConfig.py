from enum import Enum
from typing import Any, get_origin, get_args


class BaseConfig:
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
            if isinstance(value, dict):
                continue
            elif self.types[name] == str:
                data[name] = value
            elif issubclass(self.types[name], Enum):
                data[name] = None if value is None else str(value)
            elif self.types[name] == bool:
                data[name] = value
            elif self.types[name] == int:
                data[name] = value
            elif self.types[name] == float:
                if value in [float('inf'), float('-inf')]:
                    data[name] = str(value)
                else:
                    data[name] = value
            elif self.types[name] == list[str]:
                data[name] = value
            else:
                data[name] = value

        return data

    def from_dict(self, data) -> 'BaseConfig':
        for (name, _) in self.types.items():
            try:
                if issubclass(self.types[name], BaseConfig):
                    getattr(self, name).from_dict(data[name])
                elif get_origin(self.types[name]) == list:
                    list_type = get_args(self.types[name])[0]
                    list_data = data[name]
                    value = []
                    if issubclass(list_type, BaseConfig):
                        for list_entry in list_data:
                            value.append(list_type.default_values().from_dict(list_entry))
                    elif list_type == str:
                        for list_entry in list_data:
                            value.append(str(list_entry))
                    setattr(self, name, value)
                elif self.types[name] == str:
                    setattr(self, name, str(data[name]))
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
                elif self.types[name] == list[str]:
                    setattr(self, name, data[name])
                else:
                    setattr(self, name, data[name])
            except Exception:
                if name in data:
                    print(f"Could not set {name} as {str(data[name])}")
                else:
                    #print(f"Could not set {name}, not found.")
                    pass

        return self