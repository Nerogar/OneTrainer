import tkinter as tk
from enum import Enum

from typing import Any


class UIState:
    vars: dict[str, Any]

    def __init__(self, obj):
        self.obj = obj
        self.__set_vars(obj)

    def update(self, obj):
        self.obj = obj
        self.__set_vars(obj)

    def __set_str_var(self, obj, name, var):
        return lambda _0, _1, _2: setattr(obj, name, var.get())

    def __set_enum_var(self, obj, name, var, var_type):
        return lambda _0, _1, _2: setattr(obj, name, var_type[var.get()])

    def __set_bool_var(self, obj, name, var):
        return lambda _0, _1, _2: setattr(obj, name, var.get())

    def __set_int_var(self, obj, name, var):
        def update(_0, _1, _2):
            string_var = var.get()
            try:
                setattr(obj, name, float(string_var))
            except ValueError:
                pass

        return update

    def __set_float_var(self, obj, name, var):
        def update(_0, _1, _2):
            string_var = var.get()
            try:
                setattr(obj, name, int(string_var))
            except ValueError:
                pass

        return update

    def __set_vars(self, obj):
        self.vars = {}
        for name, obj_var in vars(obj).items():
            if isinstance(obj_var, str):
                var = tk.StringVar(name=name)
                var.set(obj_var)
                var.trace_add("write", self.__set_str_var(obj, name, var))
                self.vars[name] = var
            elif isinstance(obj_var, Enum):
                var = tk.StringVar(name=name)
                var.set(str(obj_var))
                var.trace_add("write", self.__set_enum_var(obj, name, var, type(obj_var)))
                self.vars[name] = var
            elif isinstance(obj_var, bool):
                var = tk.BooleanVar(name=name)
                var.set(obj_var)
                var.trace_add("write", self.__set_bool_var(obj, name, var))
                self.vars[name] = var
            elif isinstance(obj_var, int):
                var = tk.StringVar(name=name)
                var.set(str(obj_var))
                var.trace_add("write", self.__set_int_var(obj, name, var))
                self.vars[name] = var
            elif isinstance(obj_var, float):
                var = tk.StringVar(name=name)
                var.set(str(obj_var))
                var.trace_add("write", self.__set_float_var(obj, name, var))
                self.vars[name] = var
            else:
                var = tk.StringVar(name=name)
                var.set(obj_var)
                var.trace_add("write", self.__set_str_var(obj, name, var))
                self.vars[name] = var
