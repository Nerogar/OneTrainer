import contextlib
import tkinter as tk
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from modules.util.config.BaseConfig import BaseConfig
from modules.util.type_util import issubclass_safe


class UIState:
    __vars: dict[str, Any]
    __var_traces: dict[str, dict[int, Callable[[], None]]]
    __latest_var_trace_id: int

    def __init__(self, master, obj):
        self.master = master
        self.obj = obj

        self.__var_types: dict[str, type] = {}
        self.__var_nullables: dict[str, bool] = {}
        self.__var_defaults: dict[str, Any] = {}

        self.__vars = self.__create_vars(obj)
        self.__var_traces = {name: {} for name in self.__vars}
        self.__latest_var_trace_id = 0

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

    def add_var_trace(self, name, command: Callable[[], None]) -> int:
        self.__latest_var_trace_id += 1
        self.__var_traces[name][self.__latest_var_trace_id] = command
        return self.__latest_var_trace_id

    def remove_var_trace(self, name, trace_id):
        self.__var_traces[name].pop(trace_id)

    def remove_all_var_traces(self, name):
        self.__var_traces[name] = {}

    def __call_var_traces(self, name):
        for trace in self.__var_traces[name].values():
            trace()

    def __set_str_var(self, obj, is_dict, name, var, nullable):
        if is_dict:
            def update(_0, _1, _2):
                string_var = var.get()
                if (string_var == "" or string_var == "None") and nullable:
                    obj[name] = None
                else:
                    obj[name] = string_var
                self.__call_var_traces(name)
        else:
            def update(_0, _1, _2):
                string_var = var.get()
                if (string_var == "" or string_var == "None") and nullable:
                    setattr(obj, name, None)
                else:
                    setattr(obj, name, string_var)
                self.__call_var_traces(name)

        return update

    def __set_enum_var(self, obj, is_dict, name, var, var_type, nullable):
        if is_dict:
            def update(_0, _1, _2):
                string_var = var.get()
                if (string_var == "" or string_var == "None") and nullable:
                    obj[name] = None
                else:
                    obj[name] = var_type[string_var]
                self.__call_var_traces(name)
        else:
            def update(_0, _1, _2):
                string_var = var.get()
                if (string_var == "" or string_var == "None") and nullable:
                    setattr(obj, name, None)
                else:
                    setattr(obj, name, var_type[string_var])
                self.__call_var_traces(name)

        return update

    def __set_bool_var(self, obj, is_dict, name, var):
        if is_dict:
            def update(_0, _1, _2):
                obj[name] = var.get()
                self.__call_var_traces(name)
        else:
            def update(_0, _1, _2):
                setattr(obj, name, var.get())
                self.__call_var_traces(name)

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
                    with contextlib.suppress(ValueError):
                        obj[name] = int(string_var)
                self.__call_var_traces(name)
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
                    with contextlib.suppress(ValueError):
                        setattr(obj, name, int(string_var))
                self.__call_var_traces(name)

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
                    with contextlib.suppress(ValueError):
                        obj[name] = float(string_var)
                self.__call_var_traces(name)
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
                    with contextlib.suppress(ValueError):
                        setattr(obj, name, float(string_var))
                self.__call_var_traces(name)

        return update

    def __create_vars(self, obj):
        new_vars = {}

        is_dict = isinstance(obj, dict)
        is_config = isinstance(obj, BaseConfig)

        if is_config:
            for name, var_type in obj.types.items():
                self.__var_types[name] = var_type
                self.__var_nullables[name] = obj.nullables.get(name, False)
                if hasattr(obj, "default_values"):
                    self.__var_defaults[name] = obj.default_values.get(name, None)

                obj_var = getattr(obj, name)
                if issubclass_safe(var_type, BaseConfig):
                    var = UIState(self.master, obj_var)
                    new_vars[name] = var
                elif var_type is str:
                    var = tk.StringVar(master=self.master)
                    var.set("" if obj_var is None else obj_var)
                    var.trace_add("write", self.__set_str_var(obj, is_dict, name, var, obj.nullables[name]))
                    new_vars[name] = var
                elif issubclass_safe(var_type, Enum):
                    var = tk.StringVar(master=self.master)
                    var.set("" if obj_var is None else str(obj_var))
                    var.trace_add("write", self.__set_enum_var(obj, is_dict, name, var, var_type, obj.nullables[name]))
                    new_vars[name] = var
                elif var_type is bool:
                    var = tk.BooleanVar(master=self.master)
                    var.set(obj_var or False)
                    var.trace_add("write", self.__set_bool_var(obj, is_dict, name, var))
                    new_vars[name] = var
                elif var_type is int:
                    var = tk.StringVar(master=self.master)
                    var.set("" if obj_var is None else str(obj_var))
                    var.trace_add("write", self.__set_int_var(obj, is_dict, name, var, obj.nullables[name]))
                    new_vars[name] = var
                elif var_type is float:
                    var = tk.StringVar(master=self.master)
                    var.set("" if obj_var is None else str(obj_var))
                    var.trace_add("write", self.__set_float_var(obj, is_dict, name, var, obj.nullables[name]))
                    new_vars[name] = var
        else:
            iterable = obj.items() if is_dict else vars(obj).items()

            for name, obj_var in iterable:

                if isinstance(obj_var, str):
                    var = tk.StringVar(master=self.master)
                    var.set(obj_var)
                    var.trace_add("write", self.__set_str_var(obj, is_dict, name, var, False))
                    new_vars[name] = var
                elif isinstance(obj_var, Enum):
                    var = tk.StringVar(master=self.master)
                    var.set(str(obj_var))
                    var.trace_add("write", self.__set_enum_var(obj, is_dict, name, var, type(obj_var), False))
                    new_vars[name] = var
                elif isinstance(obj_var, bool):
                    var = tk.BooleanVar(master=self.master)
                    var.set(obj_var)
                    var.trace_add("write", self.__set_bool_var(obj, is_dict, name, var))
                    new_vars[name] = var
                elif isinstance(obj_var, int):
                    var = tk.StringVar(master=self.master)
                    var.set(str(obj_var))
                    var.trace_add("write", self.__set_int_var(obj, is_dict, name, var, False))
                    new_vars[name] = var
                elif isinstance(obj_var, float):
                    var = tk.StringVar(master=self.master)
                    var.set(str(obj_var))
                    var.trace_add("write", self.__set_float_var(obj, is_dict, name, var, False))
                    new_vars[name] = var

        return new_vars

    def __set_vars(self, obj):
        is_dict = isinstance(obj, dict)
        is_config = isinstance(obj, BaseConfig)
        iterable = obj.items() if is_dict else vars(obj).items()

        if is_config:
            for name, var_type in obj.types.items():
                obj_var = getattr(obj, name)
                if issubclass_safe(var_type, BaseConfig):
                    var = self.__vars[name]
                    var.__set_vars(obj_var)
                elif var_type is str:
                    var = self.__vars[name]
                    var.set("" if obj_var is None else obj_var)
                elif issubclass_safe(var_type, Enum):
                    var = self.__vars[name]
                    var.set("" if obj_var is None else str(obj_var))
                elif var_type is bool:
                    var = self.__vars[name]
                    var.set(obj_var or False)
                elif var_type in (int, float):
                    var = self.__vars[name]
                    var.set("" if obj_var is None else str(obj_var))
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
                elif isinstance(obj_var, int | float):
                    var = self.__vars[name]
                    var.set(str(obj_var))

    # metadata api
    def _resolve_state_and_leaf(self, name: str):
        parts = name.split('.')
        state: UIState = self
        for part in parts[:-1]:
            state = state.get_var(part)
            if not isinstance(state, UIState):
                return None, None
        return state, parts[-1]

    @dataclass(frozen=True)
    class VarMeta:
        type: type | None
        nullable: bool
        default: Any

    def get_field_metadata(self, name: str) -> "UIState.VarMeta":
        state, leaf = self._resolve_state_and_leaf(name)
        if state is None:
            return UIState.VarMeta(None, False, None)
        return UIState.VarMeta(
            state.__var_types.get(leaf),
            state.__var_nullables.get(leaf, False),
            state.__var_defaults.get(leaf, None),
        )
