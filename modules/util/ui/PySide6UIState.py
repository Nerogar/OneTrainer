from typing import Any

from modules.util.ui.QtVar import QtVar
from modules.util.ui.UIState import BaseUIState


class PySide6UIState(BaseUIState):
    def __init__(self, obj):
        super().__init__(obj)

    def _make_str_var(self, initial_value: Any) -> QtVar:
        return QtVar(initial_value)

    def _make_bool_var(self, initial_value: Any) -> QtVar:
        return QtVar(initial_value)

    def _make_nested_state(self, obj: Any) -> "PySide6UIState":
        return PySide6UIState(obj)
