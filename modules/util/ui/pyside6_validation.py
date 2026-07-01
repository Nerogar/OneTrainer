from collections.abc import Callable

from modules.util.enum.PathIOType import PathIOType
from modules.util.ui.QtVar import QtVar
from modules.util.ui.UIState import BaseUIState
from modules.util.ui.validation import (
    DEBOUNCE_TYPING_MS,
    ERROR_BORDER_COLOR,
    BaseFieldValidator,
    _active_validators,
    _validate_path_field,
)

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QLineEdit


class PySide6FieldValidator(BaseFieldValidator):
    def __init__(
        self,
        component: QLineEdit,
        var: QtVar,
        ui_state: BaseUIState,
        var_name: str,
        # no max_undo param: QLineEdit has native undo/redo, unlike tk.Entry
        extra_validate: Callable[[str], str | None] | None = None,
        required: bool = False,
    ):
        super().__init__(ui_state, var_name, extra_validate, required)
        self.component = component
        self.var = var
        self._original_style = component.styleSheet()
        self._syncing = False
        self._touched = False
        self._var_trace_id: int | None = None

        self._debounce = QTimer(component)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(DEBOUNCE_TYPING_MS)
        self._debounce.timeout.connect(self._on_debounce_fire)

    def _apply_error(self) -> None:
        self.component.setStyleSheet(f"border: 1px solid {ERROR_BORDER_COLOR};")

    def _clear_error(self) -> None:
        self.component.setStyleSheet(self._original_style)

    def attach(self) -> None:
        self._syncing = True
        self.component.setText(str(self.var.get()))
        self._syncing = False

        self.component.textChanged.connect(self._on_text_changed)
        self.component.editingFinished.connect(self._on_editing_finished)
        self._var_trace_id = self.var.trace_add("write", self._on_real_var_write)
        self.component.destroyed.connect(self._on_destroyed)

        self._bound = True
        _active_validators.add(self)

    def detach(self) -> None:
        if not self._bound:
            return
        self._bound = False
        _active_validators.discard(self)
        self._debounce.stop()
        self._commit()
        try:
            self.component.textChanged.disconnect(self._on_text_changed)
            self.component.editingFinished.disconnect(self._on_editing_finished)
        except RuntimeError:
            pass
        if self._var_trace_id is not None:
            self.var.trace_remove("write", self._var_trace_id)
            self._var_trace_id = None

    def _on_destroyed(self) -> None:
        """Called when the Qt C++ widget is deleted; skips _commit() since widget is gone."""
        if not self._bound:
            return
        self._bound = False
        _active_validators.discard(self)
        self._debounce.stop()
        if self._var_trace_id is not None:
            self.var.trace_remove("write", self._var_trace_id)
            self._var_trace_id = None

    def _commit(self) -> None:
        val = self.component.text()
        if val != str(self.var.get()):
            self._syncing = True
            self.var.set(val)
            self._syncing = False

    def _on_text_changed(self, _text: str) -> None:
        if self._syncing:
            return
        self._touched = True
        self._debounce.start()

    def _on_debounce_fire(self) -> None:
        val = self.component.text()
        if self._validate_and_style(val):
            self._commit()

    def _on_editing_finished(self) -> None:
        self._debounce.stop()
        if self._touched:
            val = self.component.text()
            if self._validate_and_style(val):
                self._commit()
        self._touched = False

    def _on_real_var_write(self, _0, _1, _2) -> None:
        if self._syncing:
            return
        self._syncing = True
        self.component.setText(str(self.var.get()))
        self._syncing = False
        self._validate_and_style(self.component.text())

    def flush(self) -> str | None:
        self._debounce.stop()
        val = self.component.text()
        error = self.validate(val)
        if error is not None:
            self._apply_error()
        else:
            self._clear_error()
            self._commit()
        return error


class PySide6PathValidator(PySide6FieldValidator):
    def __init__(
        self,
        component: QLineEdit,
        var: QtVar,
        ui_state: BaseUIState,
        var_name: str,
        io_type: PathIOType = PathIOType.INPUT,
        extra_validate: Callable[[str], str | None] | None = None,
        required: bool = False,
    ):
        super().__init__(component, var, ui_state, var_name,
                         extra_validate=extra_validate, required=required)
        self.io_type = io_type

    def validate(self, value: str) -> str | None:
        base_err = super().validate(value)
        if base_err is not None:
            return base_err
        if value == "":
            return None
        return _validate_path_field(self.ui_state, self.io_type, value)

    def revalidate(self) -> None:
        if self._bound:
            self._validate_and_style(self.component.text())
