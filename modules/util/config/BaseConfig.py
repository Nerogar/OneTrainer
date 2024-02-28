from enum import Enum
from typing import Any, get_origin, get_args, Callable


class BaseConfig:
    config_version: int
    config_migrations: dict[int, Callable[[dict], dict]]

    def __init__(
            self,
            data: list[(str, Any, type, bool)],
            config_version: int | None = None,
            config_migrations: dict[int, Callable[[dict], dict]] | None = None
    ):
        self.config_version = config_version if config_version is not None else 0
        self.config_migrations = config_migrations if config_migrations is not None else {}

        self.types = {}
        self.nullables = {}
        self.default_values = {}
        for (name, value, var_type, nullable) in data:
            setattr(self, name, value)
            self.types[name] = var_type
            self.nullables[name] = nullable
            self.default_values[name] = value

    def to_dict(self) -> dict:
        data = {
            '__version': self.config_version,
        }

        for (name, _) in self.types.items():
            value = getattr(self, name)
            if issubclass(self.types[name], BaseConfig):
                data[name] = value.to_dict()
            elif self.types[name] == list or get_origin(self.types[name]) == list:
                if len(get_args(self.types[name])) > 0 and issubclass(get_args(self.types[name])[0], BaseConfig):
                    if value is not None:
                        list_data = []
                        for list_entry in value:
                            list_data.append(list_entry.to_dict())
                    else:
                        list_data = None
                    data[name] = list_data
                else:
                    data[name] = value
            elif self.types[name] == dict or get_origin(self.types[name]) == dict:
                if len(get_args(self.types[name])) > 0 and issubclass(get_args(self.types[name])[1], BaseConfig):
                    dict_data = {}
                    for dict_key, dict_value in value.items():
                        dict_data[dict_key] = dict_value.to_dict()
                    data[name] = dict_data
                else:
                    data[name] = value
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

        return data

    def from_dict(self, data: dict) -> 'BaseConfig':
        version = 0
        if '__version' in data:
            version = data['__version']

        while version in self.config_migrations:
            data = self.config_migrations[version](data)
            version += 1

        for (name, _) in self.types.items():
            try:
                if issubclass(self.types[name], BaseConfig):
                    getattr(self, name).from_dict(data[name])
                elif self.types[name] == list or get_origin(self.types[name]) == list:
                    if len(get_args(self.types[name])) > 0 and issubclass(get_args(self.types[name])[0], BaseConfig):
                        list_type = get_args(self.types[name])[0]
                        if data[name] is not None:
                            old_value = getattr(self, name) if hasattr(self, name) else []
                            value = []
                            for i in range(max(len(old_value), len(data[name]))):
                                if i < len(old_value) and i < len(data[name]):
                                    value.append(old_value[i].from_dict(data[name][i]))
                                elif i < len(old_value):
                                    value.append(old_value[i])
                                else:
                                    value.append(list_type.default_values().from_dict(data[name][i]))
                        else:
                            value = None
                        setattr(self, name, value)
                    else:
                        setattr(self, name, data[name])
                elif self.types[name] == dict or get_origin(self.types[name]) == dict:
                    if len(get_args(self.types[name])) > 0 and issubclass(get_args(self.types[name])[1], BaseConfig):
                        dict_type = get_args(self.types[name])[1]
                        value = {}
                        for dict_key, dict_value in data[name].items():
                            value[dict_key] = dict_type.default_values().from_dict(dict_value)
                        setattr(self, name, value)
                    else:
                        setattr(self, name, data[name])
                elif self.types[name] == str:
                    if self.nullables[name]:
                        setattr(self, name, None if data[name] is None else str(data[name]))
                    else:
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
            except Exception:
                if name in data:
                    print(f"Could not set {name} as {str(data[name])}")
                else:
                    # print(f"Could not set {name}, not found.")
                    pass

        return self
