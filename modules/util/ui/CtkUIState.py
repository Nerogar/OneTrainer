import tkinter as tk
from typing import Any

from modules.util.ui.UIState import BaseUIState


class CtkUIState(BaseUIState):
    def __init__(self, master, obj):
        self.master = master
        super().__init__(obj)

    def _make_str_var(self, initial_value: Any):
        var = tk.StringVar(master=self.master)
        var.set(initial_value)
        return var

    def _make_bool_var(self, initial_value: Any):
        var = tk.BooleanVar(master=self.master)
        var.set(initial_value)
        return var

    def _make_nested_state(self, obj: Any) -> "CtkUIState":
        return CtkUIState(self.master, obj)
