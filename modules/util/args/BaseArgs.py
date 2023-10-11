from enum import Enum


class BaseArgs:
    def __init__(self, args: dict):
        for (key, value) in args.items():
            setattr(self, key, value)

    def to_json(self):
        data = {}
        for (key, value) in vars(self).items():
            if isinstance(value, str):
                data[key] = value
            elif isinstance(value, Enum):
                data[key] = str(value)
            elif isinstance(value, bool):
                data[key] = value
            elif isinstance(value, int):
                data[key] = value
            elif isinstance(value, float):
                data[key] = value
            elif isinstance(value, tuple) and len(value) == 2:
                data[key] = {"x": value[0], "y": value[1]}
            elif value is None:
                data[key] = "None"
            elif value in [float('inf'), float('-inf'), int('inf'), int('-inf')]:
                data[key] = str(value)
            else:
                data[key] = value

        return data

    def from_json(self, data):
        for (key, value) in vars(self).items():
            try:
                if isinstance(value, str):
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
                elif isinstance(value, tuple) and "x" in data[key] and "y" in data[key]:
                    setattr(self, key, (data[key]["x"], data[key]["y"]))
                elif data[key] == "None":
                    setattr(self, key, data[key])
                elif data[key] in ["inf", "-inf"]:
                    if isinstance(value, int):
                        setattr(self, key, int(float(data[key])))
                    else:
                        setattr(self, key, float(data[key]))
                else:
                    setattr(self, key, data[key])
            except Exception as e:
                if key in data:
                    print(f"Could not set {key} as {str(data[key])}")
                else:
                    print(f"Could not set {key}, not found.")
        return self

    def __to_arg_name(self, var_name: str) -> str:
        return "--" + var_name.replace('_', '-')

    def __to_var_name(self, arg_name: str) -> str:
        return arg_name.lstrip('-').replace('-', '_')

    def to_args(self) -> str:
        data = []
        for (key, value) in vars(self).items():
            if isinstance(value, str):
                data.append(f"{self.__to_arg_name(key)}=\"{value}\"")
            elif isinstance(value, Enum):
                data.append(f"{self.__to_arg_name(key)}=\"{str(value)}\"")
            elif isinstance(value, bool):
                if value:
                    data.append(self.__to_arg_name(key))
            elif isinstance(value, int):
                data.append(f"{self.__to_arg_name(key)}=\"{str(value)}\"")
            elif isinstance(value, float):
                data.append(f"{self.__to_arg_name(key)}=\"{str(value)}\"")
            elif value is not None:
                data.append(f"{self.__to_arg_name(key)}=\"{str(value)}\"")
            elif isinstance(value, tuple) and len(value) == 2:
                data.append(f"{self.__to_arg_name(key)}=\"({value[0]}, {value[1]})\"")
            elif value is None:
                data.append(f"{self.__to_arg_name(key)}=\"None\"")
            elif value in [float('inf'), float('-inf'), int('inf'), int('-inf')]:
                data.append(f"{self.__to_arg_name(key)}=\"{str(value)}\"")
            elif value is not None:
                data.append(f"{self.__to_arg_name(key)}=\"{str(value)}\"")

        return ' '.join(data)
