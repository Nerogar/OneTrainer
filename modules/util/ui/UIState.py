import tkinter as tk
from enum import Enum
from typing import Any

from modules.util.config.BaseConfig import BaseConfig


class UIState:
    __vars: dict[str, Any]

    def __init__(self, master, obj):
        self.master = master
        self.obj = obj
        self.__create_vars(obj)

    def update(self, obj):
        self.obj = obj
        self.__set_vars(obj)

    def get_var(self, name):
        split_name = name.split('.')

        if len(split_name) == 1:
            return self.__vars[split_name[0]]
        else:
            state = self
            for name_part in split_name:
                state = state.get_var(name_part)
            return state

    def __set_str_var(self, obj, is_dict, name, var, nullable):
        if is_dict:
            def update(_0, _1, _2):
                string_var = var.get()
                if (string_var == "" or string_var == "None") and nullable:
                    obj[name] = None
                else:
                    obj[name] = string_var
        else:
            def update(_0, _1, _2):
                string_var = var.get()
                if (string_var == "" or string_var == "None") and nullable:
                    setattr(obj, name, None)
                else:
                    setattr(obj, name, string_var)

        return update

    def __set_enum_var(self, obj, is_dict, name, var, var_type, nullable):
        if is_dict:
            def update(_0, _1, _2):
                string_var = var.get()
                if (string_var == "" or string_var == "None") and nullable:
                    obj[name] = None
                else:
                    obj[name] = var_type[string_var]
        else:
            def update(_0, _1, _2):
                string_var = var.get()
                if (string_var == "" or string_var == "None") and nullable:
                    setattr(obj, name, None)
                else:
                    setattr(obj, name, var_type[string_var])

        return update

    def __set_bool_var(self, obj, is_dict, name, var):
        if is_dict:
            def update(_0, _1, _2):
                obj[name] = var.get()
        else:
            def update(_0, _1, _2):
                setattr(obj, name, var.get())

        return update

    def __set_int_var(self, obj, is_dict, name, var, nullable):
        if is_dict:
            def update(_0, _1, _2):
                string_var = var.get()
                if (string_var == "" or string_var == "None") and nullable:
                    obj[name] = None
                elif string_var == "inf":
                    obj[name] = int("inf")
                elif string_var == "-inf":
                    obj[name] = int("-inf")
                else:
                    try:
                        obj[name] = int(string_var)
                    except ValueError:
                        pass
        else:
            def update(_0, _1, _2):
                string_var = var.get()
                if (string_var == "" or string_var == "None") and nullable:
                    setattr(obj, name, None)
                elif string_var == "inf":
                    setattr(obj, name, int("inf"))
                elif string_var == "-inf":
                    setattr(obj, name, int("-inf"))
                else:
                    try:
                        setattr(obj, name, int(string_var))
                    except ValueError:
                        pass

        return update

    def __set_float_var(self, obj, is_dict, name, var, nullable):
        if is_dict:
            def update(_0, _1, _2):
                string_var = var.get()
                if (string_var == "" or string_var == "None") and nullable:
                    obj[name] = None
                elif string_var == "inf":
                    obj[name] = float("inf")
                elif string_var == "-inf":
                    obj[name] = float("-inf")
                else:
                    try:
                        obj[name] = float(string_var)
                    except ValueError:
                        pass
        else:
            def update(_0, _1, _2):
                string_var = var.get()
                if (string_var == "" or string_var == "None") and nullable:
                    setattr(obj, name, None)
                elif string_var == "inf":
                    setattr(obj, name, float("inf"))
                elif string_var == "-inf":
                    setattr(obj, name, float("-inf"))
                else:
                    try:
                        setattr(obj, name, float(string_var))
                    except ValueError:
                        pass

        return update

    def __set_str_list_var(self, obj, is_dict, name, var):
        if is_dict:
            def update(_0, _1, _2):
                string_var = var.get()
                if string_var == "":
                    obj[name] = []
                else:
                    obj[name] = [string_var]
        else:
            def update(_0, _1, _2):
                string_var = var.get()
                if string_var == "":
                    setattr(obj, name, [])
                else:
                    setattr(obj, name, [string_var])

        return update

    def __create_vars(self, obj):
        self.__vars = {}

        is_dict = isinstance(obj, dict)
        is_config = isinstance(obj, BaseConfig)

        if is_config:
            for name, var_type in obj.types.items():
                obj_var = getattr(obj, name)
                if issubclass(var_type, BaseConfig):
                    var = UIState(self.master, obj_var)
                    self.__vars[name] = var
                elif var_type == str:
                    var = tk.StringVar(master=self.master)
                    var.set("" if obj_var is None else obj_var)
                    var.trace_add("write", self.__set_str_var(obj, is_dict, name, var, obj.nullables[name]))
                    self.__vars[name] = var
                elif issubclass(var_type, Enum):
                    var = tk.StringVar(master=self.master)
                    var.set("" if obj_var is None else str(obj_var))
                    var.trace_add("write", self.__set_enum_var(obj, is_dict, name, var, var_type, obj.nullables[name]))
                    self.__vars[name] = var
                elif var_type == bool:
                    var = tk.BooleanVar(master=self.master)
                    var.set(obj_var or False)
                    var.trace_add("write", self.__set_bool_var(obj, is_dict, name, var))
                    self.__vars[name] = var
                elif var_type == int:
                    var = tk.StringVar(master=self.master)
                    var.set("" if obj_var is None else str(obj_var))
                    var.trace_add("write", self.__set_int_var(obj, is_dict, name, var, obj.nullables[name]))
                    self.__vars[name] = var
                elif var_type == float:
                    var = tk.StringVar(master=self.master)
                    var.set("" if obj_var is None else str(obj_var))
                    var.trace_add("write", self.__set_float_var(obj, is_dict, name, var, obj.nullables[name]))
                    self.__vars[name] = var
                elif var_type == list[str]:  # TODO: think of a better solution for string lists
                    var = tk.StringVar(master=self.master)
                    var.set("" if len(obj_var) == 0 else obj_var[0])
                    var.trace_add("write", self.__set_str_list_var(obj, is_dict, name, var))
                    self.__vars[name] = var
        else:
            iterable = obj.items() if is_dict else vars(obj).items()

            for name, obj_var in iterable:
                if isinstance(obj_var, str):
                    var = tk.StringVar(master=self.master)
                    var.set(obj_var)
                    var.trace_add("write", self.__set_str_var(obj, is_dict, name, var, False))
                    self.__vars[name] = var
                elif isinstance(obj_var, Enum):
                    var = tk.StringVar(master=self.master)
                    var.set(str(obj_var))
                    var.trace_add("write", self.__set_enum_var(obj, is_dict, name, var, type(obj_var), False))
                    self.__vars[name] = var
                elif isinstance(obj_var, bool):
                    var = tk.BooleanVar(master=self.master)
                    var.set(obj_var)
                    var.trace_add("write", self.__set_bool_var(obj, is_dict, name, var))
                    self.__vars[name] = var
                elif isinstance(obj_var, int):
                    var = tk.StringVar(master=self.master)
                    var.set(str(obj_var))
                    var.trace_add("write", self.__set_int_var(obj, is_dict, name, var, False))
                    self.__vars[name] = var
                elif isinstance(obj_var, float):
                    var = tk.StringVar(master=self.master)
                    var.set(str(obj_var))
                    var.trace_add("write", self.__set_float_var(obj, is_dict, name, var, False))
                    self.__vars[name] = var
                else:
                    var = tk.StringVar(master=self.master)
                    var.set(obj_var)
                    var.trace_add("write", self.__set_str_var(obj, is_dict, name, var, False))
                    self.__vars[name] = var

    def __set_vars(self, obj):
        is_dict = isinstance(obj, dict)
        is_config = isinstance(obj, BaseConfig)
        iterable = obj.items() if is_dict else vars(obj).items()

        if is_config:
            for name, var_type in obj.types.items():
                obj_var = getattr(obj, name)
                if issubclass(var_type, BaseConfig):
                    var = self.__vars[name]
                    var.__set_vars(obj_var)
                elif var_type == str:
                    var = self.__vars[name]
                    var.set("" if obj_var is None else obj_var)
                elif issubclass(var_type, Enum):
                    var = self.__vars[name]
                    var.set("" if obj_var is None else str(obj_var))
                elif var_type == bool:
                    var = self.__vars[name]
                    var.set(obj_var or False)
                elif var_type == int:
                    var = self.__vars[name]
                    var.set("" if obj_var is None else str(obj_var))
                elif var_type == float:
                    var = self.__vars[name]
                    var.set("" if obj_var is None else str(obj_var))
                elif var_type == list[str]:
                    var = self.__vars[name]
                    var.set("" if len(obj_var) == 0 else obj_var[0])
        else:
            for name, obj_var in iterable:
                if isinstance(obj_var, str):
                    var = self.__vars[name]
                    var.set(obj_var)
                elif isinstance(obj_var, Enum):
                    var = self.__vars[name]
                    var.set(str(obj_var))
                elif isinstance(obj_var, bool):
                    var = self.__vars[name]
                    var.set(obj_var)
                elif isinstance(obj_var, int):
                    var = self.__vars[name]
                    var.set(str(obj_var))
                elif isinstance(obj_var, float):
                    var = self.__vars[name]
                    var.set(str(obj_var))
                else:
                    var = self.__vars[name]
                    var.set(obj_var)
