import tkinter as tk
from enum import Enum
from typing import Any

from modules.util.args.BaseArgs import BaseArgs


class UIState:
    vars: dict[str, Any]

    def __init__(self, master, obj):
        self.master = master
        self.obj = obj
        self.__create_vars(obj)

    def update(self, obj):
        self.obj = obj
        self.__set_vars(obj)

    def __set_str_var(self, obj, is_dict, name, var, nullable):
        if is_dict:
            def update(_0, _1, _2):
                string_var = var.get()
                if string_var == "" and nullable:
                    obj[name] = None
                else:
                    obj[name] = string_var
        else:
            def update(_0, _1, _2):

                string_var = var.get()
                if string_var == "" and nullable:
                    setattr(obj, name, None)
                else:
                    setattr(obj, name, string_var)

        return update

    def __set_enum_var(self, obj, is_dict, name, var, var_type, nullable):
        if is_dict:
            def update(_0, _1, _2):
                string_var = var.get()
                if string_var == "" and nullable:
                    obj[name] = None
                else:
                    obj[name] = var_type[string_var]
        else:
            def update(_0, _1, _2):
                string_var = var.get()
                if string_var == "" and nullable:
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
                if string_var == "" and nullable:
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
                if string_var == "" and nullable:
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
                if string_var == "" and nullable:
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
                if string_var == "" and nullable:
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

    def __set_tuple_var(self, obj, is_dict, name, var):
        if is_dict:
            def update(_0, _1, _2):
                string_var = var.get().replace('(', '').replace(')', '').replace(' ', '')
                if string_var == "" or string_var == "None":
                    obj[name] = None
                else:
                    try:
                        values = tuple(map(float, string_var.split(',')))
                        if len(values) == 2:
                            obj[name] = values
                    except ValueError:
                        pass
        else:
            def update(_0, _1, _2):
                string_var = var.get().replace('(', '').replace(')', '').replace(' ', '')
                if string_var == "" or string_var == "None":
                    setattr(obj, name, None)
                else:
                    try:
                        values = tuple(map(float, string_var.split(',')))
                        if len(values) == 2:
                            setattr(obj, name, values)
                    except ValueError:
                        pass

        return update

    def __create_vars(self, obj):
        self.vars = {}

        is_dict = isinstance(obj, dict)
        is_arg = isinstance(obj, BaseArgs)

        if is_arg:
            for name, var_type in obj.types.items():
                obj_var = getattr(obj, name)
                if var_type == str:
                    var = tk.StringVar(master=self.master)
                    var.set("" if obj_var is None else obj_var)
                    var.trace_add("write", self.__set_str_var(obj, is_dict, name, var, obj.nullables[name]))
                    self.vars[name] = var
                elif issubclass(var_type, Enum):
                    var = tk.StringVar(master=self.master)
                    var.set("" if obj_var is None else str(obj_var))
                    var.trace_add("write", self.__set_enum_var(obj, is_dict, name, var, type(obj_var), obj.nullables[name]))
                    self.vars[name] = var
                elif var_type == bool:
                    var = tk.BooleanVar(master=self.master)
                    var.set(obj_var)
                    var.trace_add("write", self.__set_bool_var(obj, is_dict, name, var))
                    self.vars[name] = var
                elif var_type == int:
                    var = tk.StringVar(master=self.master)
                    var.set("" if obj_var is None else str(obj_var))
                    var.trace_add("write", self.__set_int_var(obj, is_dict, name, var, obj.nullables[name]))
                    self.vars[name] = var
                elif var_type == float:
                    var = tk.StringVar(master=self.master)
                    var.set("" if obj_var is None else str(obj_var))
                    var.trace_add("write", self.__set_float_var(obj, is_dict, name, var, obj.nullables[name]))
                    self.vars[name] = var
                else:
                    var = tk.StringVar(master=self.master)
                    var.set("" if obj_var is None else obj_var)
                    var.trace_add("write", self.__set_str_var(obj, is_dict, name, var, obj.nullables[name]))
                    self.vars[name] = var
        else:
            iterable = obj.items() if is_dict else vars(obj).items()

            for name, obj_var in iterable:
                if isinstance(obj_var, str):
                    var = tk.StringVar(master=self.master)
                    var.set(obj_var)
                    var.trace_add("write", self.__set_str_var(obj, is_dict, name, var, False))
                    self.vars[name] = var
                elif isinstance(obj_var, Enum):
                    var = tk.StringVar(master=self.master)
                    var.set(str(obj_var))
                    var.trace_add("write", self.__set_enum_var(obj, is_dict, name, var, type(obj_var), False))
                    self.vars[name] = var
                elif isinstance(obj_var, bool):
                    var = tk.BooleanVar(master=self.master)
                    var.set(obj_var)
                    var.trace_add("write", self.__set_bool_var(obj, is_dict, name, var))
                    self.vars[name] = var
                elif isinstance(obj_var, int):
                    var = tk.StringVar(master=self.master)
                    var.set(str(obj_var))
                    var.trace_add("write", self.__set_int_var(obj, is_dict, name, var, False))
                    self.vars[name] = var
                elif isinstance(obj_var, float):
                    var = tk.StringVar(master=self.master)
                    var.set(str(obj_var))
                    var.trace_add("write", self.__set_float_var(obj, is_dict, name, var, False))
                    self.vars[name] = var
                elif isinstance(obj_var, tuple) and len(obj_var) == 2 and all(isinstance(i, float) for i in obj_var):
                    var = tk.StringVar(master=self.master)
                    var.set(str(obj_var))
                    var.trace_add("write", self.__set_tuple_var(obj, is_dict, name, var))
                    self.vars[name] = var
                else:
                    var = tk.StringVar(master=self.master)
                    var.set(obj_var)
                    var.trace_add("write", self.__set_str_var(obj, is_dict, name, var, False))
                    self.vars[name] = var

    def __set_vars(self, obj):
        is_dict = isinstance(obj, dict)
        is_arg = isinstance(obj, BaseArgs)
        iterable = obj.items() if is_dict else vars(obj).items()

        if is_arg:
            for name, var_type in obj.types.items():
                obj_var = getattr(obj, name)
                if var_type == str:
                    var = self.vars[name]
                    var.set("" if obj_var is None else obj_var)
                elif issubclass(var_type, Enum):
                    var = self.vars[name]
                    var.set("" if obj_var is None else str(obj_var))
                elif var_type == bool:
                    var = self.vars[name]
                    var.set(obj_var)
                elif var_type == int:
                    var = self.vars[name]
                    var.set("" if obj_var is None else str(obj_var))
                elif var_type == float:
                    var = self.vars[name]
                    var.set("" if obj_var is None else str(obj_var))
                elif var_type == tuple and len(obj_var) == 2 and all(isinstance(i, float) for i in obj_var):
                    var = self.vars[name]
                    var.set("" if obj_var is None else str(obj_var))
                else:
                    var = self.vars[name]
                    var.set("" if obj_var is None else obj_var)
        else:
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
                elif isinstance(obj_var, tuple) and len(obj_var) == 2 and all(isinstance(i, float) for i in obj_var):
                    var = self.vars[name]
                    var.set(str(obj_var))
                else:
                    var = self.vars[name]
                    var.set(obj_var)
