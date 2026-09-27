from collections.abc import Callable
from typing import Any


class QtVar:
    # Toolkit-neutral observable variable. Drop-in for tk.StringVar / tk.BooleanVar.

    def __init__(self, value: Any = ""):
        self._value = value
        self._traces: dict[int, Callable[[], None]] = {}
        self._next_id = 0
        self._widget_callbacks: dict[int, Callable[[Any], None]] = {}

    def get(self) -> Any:
        return self._value

    def set(self, value: Any):
        self._value = value
        for cb in list(self._widget_callbacks.values()):
            cb(value)
        for cb in list(self._traces.values()):
            cb(None, None, None)

    def trace_add(self, mode: str, callback: Callable) -> int:
        id_ = self._next_id
        self._traces[id_] = callback
        self._next_id += 1
        return id_

    def trace_remove(self, mode: str, name: int):
        self._traces.pop(name, None)

    def _bind_widget(self, push_to_widget: Callable[[Any], None]) -> int:
        # Register a one-way push from var → widget. Returns an ID for _unbind_widget.
        id_ = self._next_id
        self._widget_callbacks[id_] = push_to_widget
        self._next_id += 1
        return id_

    def _unbind_widget(self, id_: int):
        self._widget_callbacks.pop(id_, None)
