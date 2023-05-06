import tkinter as tk
from enum import Enum

from typing import Any


class UIState:
    vars: dict[str, Any]

    def __init__(self, master, obj):
        self.master = master
        self.obj = obj
        self.__create_vars(obj)

    def update(self, obj):
        self.obj = obj
        self.__set_vars(obj)

    def __set_str_var(self, obj, is_dict, name, var):
        if is_dict:
            def update(_0, _1, _2):
                obj[name] = var.get()
        else:
            def update(_0, _1, _2):
                setattr(obj, name, var.get())

        return update

    def __set_enum_var(self, obj, is_dict, name, var, var_type):
        if is_dict:
            def update(_0, _1, _2):
                obj[name] = var_type[var.get()]
        else:
            def update(_0, _1, _2):
                setattr(obj, name, var_type[var.get()])

        return update

    def __set_bool_var(self, obj, is_dict, name, var):
        if is_dict:
            def update(_0, _1, _2):
                obj[name] = var.get()
        else:
            def update(_0, _1, _2):
                setattr(obj, name, var.get())

        return update

    def __set_int_var(self, obj, is_dict, name, var):
        if is_dict:
            def update(_0, _1, _2):
                string_var = var.get()
                try:
                    obj[name] = int(string_var)
                except ValueError:
                    pass
        else:
            def update(_0, _1, _2):
                string_var = var.get()
                try:
                    setattr(obj, name, int(string_var))
                except ValueError:
                    pass

        return update

    def __set_float_var(self, obj, is_dict, name, var):
        if is_dict:
            def update(_0, _1, _2):
                string_var = var.get()
                try:
                    obj[name] = float(string_var)
                except ValueError:
                    pass
        else:
            def update(_0, _1, _2):
                string_var = var.get()
                try:
                    setattr(obj, name, float(string_var))
                except ValueError:
                    pass

        return update

    def __create_vars(self, obj):
        self.vars = {}

        is_dict = isinstance(obj, dict)
        iterable = obj.items() if is_dict else vars(obj).items()

        for name, obj_var in iterable:
            if isinstance(obj_var, str):
                var = tk.StringVar(master=self.master)
                var.set(obj_var)
                var.trace_add("write", self.__set_str_var(obj, is_dict, name, var))
                self.vars[name] = var
            elif isinstance(obj_var, Enum):
                var = tk.StringVar(master=self.master)
                var.set(str(obj_var))
                var.trace_add("write", self.__set_enum_var(obj, is_dict, name, var, type(obj_var)))
                self.vars[name] = var
            elif isinstance(obj_var, bool):
                var = tk.BooleanVar(master=self.master)
                var.set(obj_var)
                var.trace_add("write", self.__set_bool_var(obj, is_dict, name, var))
                self.vars[name] = var
            elif isinstance(obj_var, int):
                var = tk.StringVar(master=self.master)
                var.set(str(obj_var))
                var.trace_add("write", self.__set_int_var(obj, is_dict, name, var))
                self.vars[name] = var
            elif isinstance(obj_var, float):
                var = tk.StringVar(master=self.master)
                var.set(str(obj_var))
                var.trace_add("write", self.__set_float_var(obj, is_dict, name, var))
                self.vars[name] = var
            else:
                var = tk.StringVar(master=self.master)
                var.set(obj_var)
                var.trace_add("write", self.__set_str_var(obj, is_dict, name, var))
                self.vars[name] = var

    def __set_vars(self, obj):
        is_dict = isinstance(obj, dict)
        iterable = obj.items() if is_dict else vars(obj).items()

        for name, obj_var in iterable:
            if isinstance(obj_var, str):
                var = self.vars[name]
                var.set(obj_var)
            elif isinstance(obj_var, Enum):
                var = self.vars[name]
                var.set(str(obj_var))
            elif isinstance(obj_var, bool):
                var = self.vars[name]
                var.set(obj_var)
            elif isinstance(obj_var, int):
                var = self.vars[name]
                var.set(str(obj_var))
            elif isinstance(obj_var, float):
                var = self.vars[name]
                var.set(str(obj_var))
            else:
                var = self.vars[name]
                var.set(obj_var)
