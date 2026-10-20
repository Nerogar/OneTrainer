import html
import re
from collections.abc import Callable
from pathlib import PurePosixPath

from modules.util.config.TrainConfig import get_output_model_destination, is_auto_run_name_mode, prepare_run_name
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.PathIOType import PathIOType
from modules.util.path_util import supported_image_extensions, supported_video_extensions
from modules.util.ui.autocorrect import autocorrect_path
from modules.util.ui.QtVar import QtVar
from modules.util.ui.UIState import BaseUIState
from modules.util.ui.validation import (
    ERROR_BORDER_COLOR,
    BaseFieldValidator,
    _active_validators,
    _check_overwrite,
    validate_path,
)

from PySide6.QtCore import QPoint, QTimer
from PySide6.QtWidgets import QLineEdit, QToolTip

_RUN_NAME_EXTENSIONS = {".safetensors", ".ckpt", ".pt", ".bin"} | supported_image_extensions() | supported_video_extensions()


class PySide6FieldValidator(BaseFieldValidator):
    def __init__(
        self,
        component: QLineEdit,
        var: QtVar,
        ui_state: BaseUIState,
        var_name: str,
        extra_validate: Callable[[str], str | None] | None = None,
        required: bool = False,
    ):
        super().__init__(ui_state, var_name, extra_validate, required)
        self.component = component
        self.var = var
        self._original_style = component.styleSheet()
        self._original_tooltip = component.toolTip()
        self._syncing = False
        self._touched = False
        self._cleaned_value: str | None = None
        self._traces: list[tuple[QtVar, int]] = []
        self._debounce = QTimer(component)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(500)
        self._debounce.timeout.connect(self._on_debounce_fire)

    def _get_value(self, name: str, default=None):
        var = self._get_var_safe(name)
        return var.get() if var is not None else default

    def _trace(self, name: str, callback) -> None:
        var = self._get_var_safe(name)
        if var is not None:
            self._traces.append((var, var.trace_add("write", callback)))

    def _apply_error(self) -> None:
        error = self.validate(self.component.text())
        self.component.setStyleSheet(f"{self._original_style}\nborder: 1px solid {ERROR_BORDER_COLOR};")
        self.component.setToolTip(f"<p>{html.escape(error or 'Invalid value')}</p>")
        if self.component.hasFocus():
            QToolTip.showText(self.component.mapToGlobal(QPoint(0, self.component.height())),
                              self.component.toolTip(), self.component, msecShowTime=5000)

    def _clear_error(self) -> None:
        if self.component.toolTip() != self._original_tooltip:
            QToolTip.hideText()
        self.component.setStyleSheet(self._original_style)
        self.component.setToolTip(self._original_tooltip)

    def attach(self) -> None:
        self.component.setText(str(self.var.get()))
        self.component.textChanged.connect(self._on_text_changed)
        self.component.editingFinished.connect(self._on_editing_finished)
        self._trace(self.var_name, self._on_real_var_write)
        self.component.destroyed.connect(self._on_destroyed)
        self._bound = True
        _active_validators.add(self)

    def _on_destroyed(self) -> None:
        self._bound = False
        _active_validators.discard(self)
        for var, trace_id in self._traces:
            var.trace_remove("write", trace_id)
        self._traces.clear()

    def detach(self) -> None:
        if not self._bound:
            return
        self._debounce.stop()
        if self.validate(self.component.text()) is None:
            self._commit()
        self.component.textChanged.disconnect(self._on_text_changed)
        self.component.editingFinished.disconnect(self._on_editing_finished)
        self._on_destroyed()

    def _commit(self) -> None:
        value = self.component.text()
        if value != str(self.var.get()):
            self._syncing = True
            try:
                self.var.set(value)
            finally:
                self._syncing = False

    def _replace_text(self, value: str) -> None:
        # Unlike setText(), an edit preserves QLineEdit's native undo history.
        self._syncing = True
        try:
            self.component.selectAll()
            self.component.insert(value)
        finally:
            self._syncing = False

    def _cleanup(self, value: str) -> str:
        return value.strip()

    def _prepare_commit(self) -> str:
        value = self.component.text()
        if self._get_value("auto_correct_input", False) and value != self._cleaned_value:
            cleaned = self._cleanup(value)
            if cleaned != value:
                self._replace_text(cleaned)
            self._cleaned_value = cleaned
            return cleaned
        return value

    def _on_text_changed(self, _text: str) -> None:
        if not self._syncing:
            self._cleaned_value = None
            self._touched = True
            self._debounce.start()

    def _on_debounce_fire(self) -> None:
        if self._validate_and_style(self.component.text()):
            self._commit()

    def _on_editing_finished(self) -> None:
        if self._touched:
            self.flush()
        self._touched = False

    def _on_real_var_write(self, *_args) -> None:
        if self._syncing:
            return
        self._debounce.stop()
        self._cleaned_value = None
        self._touched = False
        self._syncing = True
        self.component.setText(str(self.var.get()))
        self._syncing = False
        self.revalidate()

    def revalidate(self, *_args) -> None:
        if self._bound:
            self._validate_and_style(self.component.text())

    def flush(self) -> str | None:
        self._debounce.stop()
        value = self._prepare_commit()
        error = self.validate(value)
        if error is None:
            self._clear_error()
            self._commit()
        else:
            self._apply_error()
        return error


class PySide6PathValidator(PySide6FieldValidator):
    def __init__(self, *args, io_type: PathIOType = PathIOType.INPUT, output_is_dir: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.io_type = io_type
        self.output_is_dir = output_is_dir

    def _cleanup(self, value: str) -> str:
        return autocorrect_path(value, self.io_type)

    def validate(self, value: str) -> str | None:
        error = super().validate(value)
        if error is not None or value == "":
            return error
        return validate_path(value, self.io_type, output_is_dir=self.output_is_dir,
                             prevent_overwrites=self._get_value("prevent_overwrites", False),
                             output_format=self._get_value("output_model_format"))


class PySide6RunNameValidator(PySide6FieldValidator):
    def attach(self) -> None:
        super().attach()
        self._blank_timer = QTimer(self.component)
        self._blank_timer.setSingleShot(True)
        self._blank_timer.setInterval(10_000)
        self._blank_timer.timeout.connect(self.flush)
        self.component.textChanged.connect(self._schedule_blank_fill)
        for name in ("final_output_dir", "output_model_format", "prevent_overwrites"):
            self._trace(name, self.revalidate)
        self._trace("run_name_mode", self._on_mode_change)
        self.component.setReadOnly(is_auto_run_name_mode(self._get_value("run_name_mode")))
        if not self.component.text().strip():
            self.flush()

    def detach(self) -> None:
        self._blank_timer.stop()
        self.component.textChanged.disconnect(self._schedule_blank_fill)
        super().detach()

    def _schedule_blank_fill(self, value: str) -> None:
        self._blank_timer.stop()
        if not self._syncing and not value.strip():
            self._blank_timer.start()

    def _on_mode_change(self, *_args) -> None:
        automatic = is_auto_run_name_mode(self._get_value("run_name_mode"))
        self.component.setReadOnly(automatic)
        if self.ui_state.updating:
            return
        if automatic:
            self._replace_text(prepare_run_name(self._get_value("training_method"),
                                              self._get_value("run_name_mode"), is_new_invocation=True))
        self.flush()

    def _prepare_commit(self) -> str:
        self._blank_timer.stop()
        if not self.component.text().strip():
            self._replace_text(prepare_run_name(self._get_value("training_method"), self._get_value("run_name_mode")))
        return super()._prepare_commit()

    def validate(self, value: str) -> str | None:
        error = super().validate(value)
        if error is not None or not value:
            return error
        if re.search(r"[/\\]", value):
            return "Run name must not contain path separators (/ or \\)"
        if re.search(r"[^\w .\-]", value) or value in (".", ".."):
            return "Run name contains invalid characters"
        if PurePosixPath(value).suffix.lower() in _RUN_NAME_EXTENSIONS:
            return "Run name must not end with a file extension"
        output_format = ModelFormat[str(self._get_value("output_model_format", "SAFETENSORS"))]
        destination = get_output_model_destination(self._get_value("final_output_dir", ""), value, output_format)
        return _check_overwrite(destination, is_dir=not output_format.is_single_file(),
                                prevent=self._get_value("prevent_overwrites", False))
