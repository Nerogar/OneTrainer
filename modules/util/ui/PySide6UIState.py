from typing import Any

from modules.util.ui.QtVar import QtVar
from modules.util.ui.UIState import BaseUIState


class PySide6UIState(BaseUIState):
    def __init__(self, obj):
        self.updating = False
        super().__init__(obj)

    def update(self, obj):
        self.updating = True
        try:
            super().update(obj)
        finally:
            self.updating = False

    def _make_str_var(self, initial_value: Any) -> QtVar:
        return QtVar(initial_value)

    def _make_bool_var(self, initial_value: Any) -> QtVar:
        return QtVar(initial_value)

    def _make_nested_state(self, obj: Any) -> "PySide6UIState":
        return PySide6UIState(obj)
