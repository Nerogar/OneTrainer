from enum import Enum


class BaseParams:
    def __init__(self, args: dict):
        for (key, value) in args.items():
            setattr(self, key, value)

    def to_dict(self):
        data = {}
        for (key, value) in vars(self).items():
            if isinstance(value, dict):
                continue
            elif isinstance(value, str):
                data[key] = value
            elif isinstance(value, Enum):
                data[key] = str(value)
            elif isinstance(value, bool):
                data[key] = value
            elif isinstance(value, int):
                data[key] = value
            elif isinstance(value, float):
                data[key] = value
            else:
                data[key] = value

        return data

    def from_dict(self, data):
        for (key, value) in vars(self).items():
            try:
                if isinstance(value, BaseParams):
                    continue
                elif isinstance(value, str):
                    setattr(self, key, data[key])
                elif isinstance(value, Enum):
                    enum_type = type(getattr(self, key))
                    setattr(self, key, enum_type[data[key]])
                elif isinstance(value, bool):
                    setattr(self, key, data[key])
                elif isinstance(value, int):
                    setattr(self, key, int(data[key]))
                elif isinstance(value, float):
                    setattr(self, key, float(data[key]))
                else:
                    setattr(self, key, data[key])
            except Exception as e:
                if key in data:
                    print(f"Could not set {key} as {str(data[key])}")
                else:
                    #print(f"Could not set {key}, not found.")
                    pass

        return self