import contextlib
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from modules.util.enum.PathIOType import PathIOType
from modules.util.enum.TimeUnit import TimeUnit
from modules.util.path_util import supported_image_extensions, supported_video_extensions
from modules.util.ui.pyside6_util import format_tooltip, limit_input_width
from modules.util.ui.pyside6_validation import PySide6FieldValidator, PySide6PathValidator, PySide6RunNameValidator
from modules.util.ui.QtQuickSwitch import QtQuickSwitch
from modules.util.ui.QtTextBox import QtTextBox, QtTextPreview
from modules.util.ui.UIState import BaseUIState

from PySide6.QtCore import (
    QAbstractAnimation,
    QEasingCurve,
    QEvent,
    QObject,
    QRectF,
    QSignalBlocker,
    QSize,
    Qt,
    QTimer,
    QVariantAnimation,
    Signal,
)
from PySide6.QtGui import QFont, QFontMetricsF, QPalette, QPixmap, QWheelEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStyle,
    QStyleOptionButton,
    QStyleOptionComboBox,
    QStylePainter,
    QToolButton,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

PAD = 10
FORM_ROW_SPACING = PAD + 4
PATH_ENTRY_GAP = 8


def _set_max_width(
        widget: QWidget,
        max_width: int | None,
        *,
        text_widget: QWidget | None = None,
        extra_width: int = 0,
) -> None:
    """Cap an input in characters, adding accessory space for compound controls."""
    if max_width is None:
        return
    target = text_widget if text_widget is not None else widget
    limit_input_width(target, max_width)
    if target is not widget:
        pixels = target.maximumWidth() + extra_width
        widget.setMinimumWidth(min(widget.minimumWidth(), pixels))
        widget.setMaximumWidth(pixels)
    # A capped input should stay beside its label, not float in its grid cell.
    layout = widget.parentWidget().layout()
    item = layout.itemAt(layout.indexOf(widget)) if layout is not None else None
    if item is not None and not item.alignment() & Qt.AlignmentFlag.AlignHorizontal_Mask:
        layout.setAlignment(widget, item.alignment() | Qt.AlignmentFlag.AlignLeft)


# ---------------------------------------------------------------------------
# PySide6-only helpers
# ---------------------------------------------------------------------------

class CollapsibleForm(QWidget):
    """Reveal an intact form by clipping its height, preserving values and bindings."""

    transition_finished = Signal()

    def __init__(self, parent):
        super().__init__(parent)
        self.content = QWidget(self)
        self.content.setEnabled(False)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.expanded = False
        self._animation = QVariantAnimation(self)
        self._animation.setDuration(180)
        self._animation.setEasingCurve(QEasingCurve.Type.InOutCubic)
        self._animation.valueChanged.connect(self.setFixedHeight)
        self._animation.finished.connect(self._finish)
        self.setFixedHeight(0)
        self.hide()

    @property
    def is_animating(self) -> bool:
        return self._animation.state() == QAbstractAnimation.State.Running

    def sizeHint(self):
        return self.content.sizeHint()

    def minimumSizeHint(self):
        return QSize(self.content.minimumSizeHint().width(), 0)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.content.setGeometry(0, 0, self.width(), self.content.sizeHint().height())

    def set_expanded(self, expanded: bool, *, animate: bool = True) -> None:
        if expanded == self.expanded:
            if not animate:
                self._animation.stop()
                self._finish()
            return
        self.expanded = expanded
        height = self.height() if not self.isHidden() else 0
        self._animation.stop()
        self.content.setEnabled(expanded)
        if not animate:
            self._finish()
            return
        self.show()
        self._animation.setStartValue(height)
        self._animation.setEndValue(self.content.sizeHint().height() if expanded else 0)
        self._animation.start()

    def _finish(self):
        if self.expanded:
            self.setMinimumHeight(0)
            self.setMaximumHeight(16777215)
            self.show()
        else:
            self.setFixedHeight(0)
            self.hide()
        self.updateGeometry()
        self.transition_finished.emit()


class _DisabledToolTipFilter(QObject):
    def __init__(self, control: QWidget, *widgets: QWidget):
        super().__init__(control)
        for widget in (control, *widgets):
            widget.installEventFilter(self)

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.ToolTip and not self.parent().isEnabled() and watched.toolTip():
            QToolTip.showText(event.globalPos(), watched.toolTip(), watched)
            return True
        return super().eventFilter(watched, event)


class _FileDropFilter(QObject):
    def __init__(
            self,
            widget: QWidget,
            command: Callable[[list[str]], None],
            path_filter: Callable[[Path], bool],
            multiple: bool = False,
    ):
        super().__init__(widget)
        self.command = command
        self.path_filter = path_filter
        self.multiple = multiple
        widget.setAcceptDrops(True)
        widget.installEventFilter(self)

    def _set_hover(self, widget, hovered):
        if isinstance(widget, QLineEdit) and bool(widget.property("drop_hover")) != hovered:
            widget.setProperty("drop_hover", hovered)
            widget.style().unpolish(widget)
            widget.style().polish(widget)
            widget.update()

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.DragLeave:
            self._set_hover(watched, False)
            return False
        if event.type() not in (QEvent.Type.DragEnter, QEvent.Type.DragMove, QEvent.Type.Drop):
            return False
        if not event.mimeData().hasUrls():
            self._set_hover(watched, False)
            return False

        event.ignore()
        if not watched.isEnabled() or isinstance(watched, QLineEdit) and watched.isReadOnly():
            self._set_hover(watched, False)
            return True
        if not event.possibleActions() & Qt.DropAction.CopyAction:
            self._set_hover(watched, False)
            return True

        urls = event.mimeData().urls()
        if not self.multiple and len(urls) != 1:
            self._set_hover(watched, False)
            return True

        paths = []
        for url in urls:
            if url.isLocalFile():
                path = url.toLocalFile()
                with contextlib.suppress(OSError):
                    if path and self.path_filter(Path(path)):
                        paths.append(path)
        self._set_hover(watched, bool(paths) and event.type() != QEvent.Type.Drop)
        if paths:
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
            if event.type() == QEvent.Type.Drop:
                self.command(paths)
        return True


def _layout(master: QWidget) -> QGridLayout:
    lo = master.layout()
    if lo is None:
        lo = QGridLayout(master)
        lo.setContentsMargins(0, 0, 0, 0)
        lo.setHorizontalSpacing(PAD)
        lo.setVerticalSpacing(FORM_ROW_SPACING)
        master.setLayout(lo)
    return lo


def _set_tooltip(component: QWidget, text: str, wide: bool = False) -> None:
    component.setToolTip(format_tooltip(text))


def _alignment(sticky: str) -> Qt.AlignmentFlag:
    has_e = 'e' in sticky
    has_w = 'w' in sticky
    has_n = 'n' in sticky
    has_s = 's' in sticky

    if has_e and has_w:
        h = Qt.AlignmentFlag(0)
    elif has_e:
        h = Qt.AlignRight
    else:
        h = Qt.AlignLeft

    has_v = 'v' in sticky
    if has_n and has_s:
        v = Qt.AlignmentFlag(0)
    elif has_s:
        v = Qt.AlignBottom
    elif has_v:
        v = Qt.AlignVCenter
    else:
        v = Qt.AlignTop

    return h | v


def _add(
        layout: QGridLayout,
        widget: QWidget,
        row: int,
        col: int,
        sticky: str = "new",
        padx: int = PAD,
        pady: int = PAD,
        rowspan: int = 1,
        colspan: int = 1,
):
    layout.addWidget(widget, row, col, rowspan, colspan)
    label_item = layout.itemAtPosition(row, col - 1) if col else None
    if label_item is not None and isinstance(label_item.widget(), QLabel):
        widget.setAccessibleName(label_item.widget().text())
    align = _alignment(sticky)
    if align:
        layout.setAlignment(widget, align)


def scrollable_frame(parent: QWidget) -> tuple[QScrollArea, QWidget]:
    scroll = QScrollArea(parent)
    scroll.setWidgetResizable(True)
    container = QWidget()
    container_layout = QVBoxLayout(container)
    container_layout.setContentsMargins(PAD, PAD, PAD, PAD)
    container_layout.setSpacing(0)
    frame = QWidget(container)
    container_layout.addWidget(frame)
    container_layout.addStretch(1)
    scroll.setWidget(container)
    return scroll, frame


def _pack_form(master: QWidget) -> None:
    # Add a stretch row and column after the last content cell so extra space
    # goes to the empty gutter rather than stretching content widgets. Skip this
    # if a content row/column already claims stretch (e.g. an entry meant to
    # grow) - adding another stretchy gutter would only split the extra space
    # between the two instead of giving it all to the intended one.
    lo = _layout(master)
    if not any(lo.rowStretch(r) for r in range(lo.rowCount())):
        lo.setRowStretch(lo.rowCount(), 1)
    if not any(lo.columnStretch(c) for c in range(lo.columnCount())):
        lo.setColumnStretch(lo.columnCount(), 1)


# ---------------------------------------------------------------------------
# Stateless widgets
# ---------------------------------------------------------------------------

def app_title(master: QWidget, row: int, column: int):
    frame = QFrame(master)
    layout = QGridLayout(frame)
    layout.setContentsMargins(5, 5, 5, 5)
    _layout(master).addWidget(frame, row, column)

    pixmap = QPixmap("resources/icons/icon.png").scaled(
        40, 40, Qt.KeepAspectRatio, Qt.SmoothTransformation
    )
    icon_label = QLabel(frame)
    icon_label.setPixmap(pixmap)
    layout.addWidget(icon_label, 0, 0)

    text_label = QLabel("OneTrainer", frame)
    font = text_label.font()
    font.setPointSize(14)
    font.setBold(True)
    text_label.setFont(font)
    layout.addWidget(text_label, 0, 1)


def label(
        master: QWidget,
        row: int,
        column: int,
        text: str,
        pad: int = PAD,
        tooltip: str | None = None,
        wide_tooltip: bool = False,
        wraplength: int = 0,
        underline: bool = False,
        *,
        heading: bool = False,
) -> QLabel:
    component = QLabel(text, master)
    component.setProperty("formLabel", "heading" if heading else "field")
    font = master.font()
    font.setPointSizeF(max(10.0, font.pointSizeF()) + int(heading))
    font.setWeight(QFont.Weight.Bold if heading else QFont.Weight.Normal)
    if underline:
        font.setUnderline(True)
    component.setFont(font)
    cell_alignment = Qt.AlignVCenter | Qt.AlignLeft
    if wraplength > 0:
        component.setWordWrap(True)
        component.setMaximumWidth(wraplength)
        # multi-line labels must not be vertically centered: if a neighboring
        # widget in the same row ever forces the row shorter than this label's
        # wrapped text, centering clips the top and bottom lines, leaving only
        # the middle line visible
        component.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        component.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        cell_alignment = Qt.AlignTop | Qt.AlignLeft
    if tooltip:
        _set_tooltip(component, tooltip, wide_tooltip)
    layout = _layout(master)
    layout.addWidget(component, row, column)
    layout.setAlignment(component, cell_alignment)
    return component


def header_label(
        master: QWidget,
        row: int,
        column: int,
        text: str,
        *,
        tooltip: str | None = None,
        top_spacing: int = 0,
) -> QLabel:
    component = label(master, row, column, text, tooltip=tooltip, heading=True)
    component.setContentsMargins(0, top_spacing, 0, 0)
    return component


# ---------------------------------------------------------------------------
# Compound widgets
# ---------------------------------------------------------------------------

def labeled_field(master: QWidget, ui_state: BaseUIState, row: int, factory: Callable, var_name: str,
                  title: str, help_text: str, **kwargs):
    label(master, row, 0, title, tooltip=help_text)
    return factory(master, row, 1, ui_state=ui_state, var_name=var_name, **kwargs)


def entry(
        master: QWidget,
        row: int,
        column: int,
        ui_state: BaseUIState,
        var_name: str,
        command: Callable[[], None] | None = None,
        tooltip: str = "",
        wide_tooltip: bool = False,
        width: int = 140,
        sticky: str = "new",
        max_undo: int | None = None,  # unused: kept for signature parity with ctk_components.entry()
        validator_factory: Callable[..., PySide6FieldValidator] | None = None,
        extra_validate: Callable[[str], str | None] | None = None,
        required: bool = False,
        placeholder: str = "",
        allow_empty: bool = False,
        *,
        max_width: int | None = None,
) -> QLineEdit:
    """Create a text input; width is a pixel minimum, max_width a character cap."""
    var = ui_state.get_var(var_name)

    component = QLineEdit(master)
    component.setMinimumWidth(width)
    if placeholder:
        component.setPlaceholderText(placeholder)
    _add(_layout(master), component, row, column, sticky=sticky)

    if command:
        trace_id = ui_state.add_var_trace(var_name, command)
        component.destroyed.connect(lambda: ui_state.remove_var_trace(var_name, trace_id))

    if tooltip:
        _set_tooltip(component, tooltip, wide_tooltip)

    if validator_factory is None and var_name == "run_name":
        validator_factory = PySide6RunNameValidator

    if validator_factory is not None:
        validator = validator_factory(
            component, var, ui_state, var_name,
            extra_validate=extra_validate,
            required=required,
            allow_empty=allow_empty,
        )
    else:
        validator = PySide6FieldValidator(
            component, var, ui_state, var_name,
            extra_validate=extra_validate,
            required=required,
            allow_empty=allow_empty,
        )
    validator.display_name = component.accessibleName() or master.accessibleName() or var_name
    validator.attach()
    component._validator = validator  # type: ignore[attr-defined]

    _set_max_width(component, max_width)
    return component


def text_box(
        master: QWidget,
        row: int,
        column: int,
        ui_state: BaseUIState,
        var_name: str,
        *,
        lines: int = 2,
        resizable: bool = False,
        max_lines: int | None = None,
) -> QtTextBox:
    component = QtTextBox(master, lines=lines, resizable=resizable, max_lines=max_lines)
    component.setMinimumWidth(140)
    _add(_layout(master), component, row, column)
    var = ui_state.get_var(var_name)
    component.setPlainText(str(var.get()))

    def update_text(value):
        # Avoid resetting the cursor and undo history when this editor writes.
        if component.toPlainText() != str(value):
            with QSignalBlocker(component):
                component.setPlainText(str(value))

    binding = var._bind_widget(update_text)
    component.destroyed.connect(lambda: var._unbind_widget(binding))
    component.textChanged.connect(lambda: var.set(component.toPlainText()))
    return component


def text_preview(
        master: QWidget,
        row: int,
        column: int,
        ui_state: BaseUIState,
        var_name: str,
        *,
        lines: int = 2,
        on_click: Callable[[], None] | None = None,
        tooltip: str | None = None,
) -> QtTextPreview:
    component = QtTextPreview(master, lines=lines, clickable=on_click is not None)
    component.setMinimumWidth(140)
    _add(_layout(master), component, row, column)
    if on_click is not None:
        component.clicked.connect(on_click)
    if tooltip:
        _set_tooltip(component, tooltip)
    var = ui_state.get_var(var_name)
    component.setText(str(var.get()))
    binding = var._bind_widget(lambda value: component.setText(str(value)))
    component.destroyed.connect(lambda: var._unbind_widget(binding))
    return component


def path_entry(
        master: QWidget,
        row: int,
        column: int,
        ui_state: BaseUIState,
        var_name: str,
        *,
        mode: Literal["file", "dir"] = "file",
        io_type: PathIOType = PathIOType.INPUT,
        path_modifier: Callable[[str], str | Path] | None = None,
        allow_model_files: bool = True,
        allow_image_files: bool = False,
        allow_video_files: bool = False,
        command: Callable[[str], None] | None = None,
        extra_validate: Callable[[str], str | None] | None = None,
        required: bool = False,
        columnspan: int = 1,
        placeholder: str = "",
        max_width: int | None = None,
) -> QWidget:
    """Cap the path text in characters with max_width; add space for the browse button."""
    frame = QWidget(master)
    frame_lo = QGridLayout(frame)
    frame_lo.setContentsMargins(0, 0, 0, 0)
    frame_lo.setSpacing(0)
    frame_lo.setHorizontalSpacing(PATH_ENTRY_GAP)
    frame_lo.setColumnStretch(0, 1)
    _add(_layout(master), frame, row, column, sticky="new", padx=0, pady=0, colspan=columnspan)

    def _path_validator_factory(comp, var, state, name, **kw):
        comp.setStyleSheet('QLineEdit[drop_hover="true"] { border: 1px solid #3b82f6; }')
        return PySide6PathValidator(comp, var, state, name, io_type=io_type, output_is_dir=mode == "dir", **kw)

    entry_component = entry(
        frame, 0, 0, ui_state, var_name,
        validator_factory=_path_validator_factory,
        extra_validate=extra_validate,
        required=required,
        placeholder=placeholder,
    )

    dep_trace_ids: list[tuple] = []
    if io_type in (PathIOType.OUTPUT, PathIOType.MODEL):
        validator = getattr(entry_component, '_validator', None)
        if validator is not None:
            for dep_var_name in ("prevent_overwrites", "output_model_format"):
                with contextlib.suppress(KeyError, AttributeError):
                    dep_var = ui_state.get_var(dep_var_name)
                    tid = dep_var.trace_add("write", lambda _0, _1, _2: validator.revalidate())
                    dep_trace_ids.append((dep_var, tid))

    if dep_trace_ids:
        def _cleanup_dep_traces():
            for dv, tid in dep_trace_ids:
                dv.trace_remove("write", tid)
        frame.destroyed.connect(_cleanup_dep_traces)

    use_save_dialog = io_type in (PathIOType.OUTPUT, PathIOType.MODEL)

    def _set_path(chosen):
        if path_modifier:
            chosen = path_modifier(chosen)
        chosen_str = str(chosen)
        entry_component.selectAll()
        entry_component.insert(chosen_str)
        entry_component._validator._commit()
        entry_component._validator.revalidate()
        if command:
            command(chosen_str)
        entry_component.window().activateWindow()
        entry_component.setFocus(Qt.MouseFocusReason)
        entry_component.end(False)

    def _accept_path(path):
        if mode == "dir":
            return path.is_dir()
        return path.is_file() or (path_modifier is not None or io_type == PathIOType.MODEL) and path.is_dir()

    entry_component._drop_filter = _FileDropFilter(
        entry_component, lambda paths: _set_path(paths[0]), _accept_path,
    )

    def _open_dialog():
        current_path_str = ui_state.get_var(var_name).get() or None
        current_dir = ""
        current_filename = ""

        if current_path_str:
            current_path = Path(current_path_str)
            if mode == "file":
                current_dir = str(current_path.parent)
                current_filename = str(current_path.name)
            elif mode == "dir":
                current_dir = str(current_path.parent)

        if mode == "dir":
            chosen = QFileDialog.getExistingDirectory(frame, "", current_dir, QFileDialog.Option.ShowDirsOnly)
        else:
            filters = ["All Files (*.*)"]
            if allow_model_files:
                filters += [
                    "Diffusers (model_index.json)",
                    "Checkpoint (*.ckpt *.pt *.bin)",
                    "Safetensors (*.safetensors)",
                ]
            if allow_image_files:
                exts = " ".join(f"*.{x}" for x in supported_image_extensions())
                filters.append(f"Image ({exts})")
            if allow_video_files:
                exts = " ".join(f"*{e}" for e in supported_video_extensions())
                filters.append(f"Video ({exts})")
            filter_str = ";;".join(filters)
            init_path = str(Path(current_dir) / current_filename) if current_filename else current_dir

            if use_save_dialog:
                chosen, _ = QFileDialog.getSaveFileName(frame, "", init_path, filter_str)
            else:
                chosen, _ = QFileDialog.getOpenFileName(frame, "", init_path, filter_str)

        if chosen:
            _set_path(chosen)

    btn = QPushButton("...", frame)
    btn.setFixedWidth(40)
    btn.clicked.connect(_open_dialog)
    frame_lo.addWidget(btn, 0, 1)

    _set_max_width(frame, max_width, text_widget=entry_component, extra_width=btn.width() + PATH_ENTRY_GAP)
    return frame


def time_entry(
        master: QWidget,
        row: int,
        column: int,
        ui_state: BaseUIState,
        var_name: str,
        unit_var_name: str,
        supports_time_units: bool = True,
        *,
        max_width: int | None = None,
) -> QWidget:
    """Cap the numeric input in characters with max_width; add space for the units."""
    frame = QWidget(master)
    _add(_layout(master), frame, row, column, sticky="new", padx=0, pady=0)

    value_entry = entry(frame, 0, 0, ui_state, var_name, width=50)

    values = [str(x) for x in list(TimeUnit)]
    if not supports_time_units:
        values = [str(x) for x in list(TimeUnit) if not x.is_time_unit()]

    unit_combo = options(frame, 0, 1, values, ui_state, unit_var_name)

    unit_combo.ensurePolished()
    _set_max_width(frame, max_width, text_widget=value_entry,
                   extra_width=unit_combo.sizeHint().width() + _layout(frame).horizontalSpacing())
    return frame


def layer_filter_entry(
        master: QWidget,
        row: int,
        column: int,
        ui_state: BaseUIState,
        preset_var_name: str,
        preset_label: str,
        preset_tooltip: str,
        presets,
        entry_var_name: str,
        entry_tooltip: str,
        regex_var_name: str,
        regex_tooltip: str,
) -> QWidget:
    frame = QFrame(master)
    frame.setProperty("formInset", True)
    _layout(master).addWidget(frame, row, column)
    layout = _layout(frame)
    layout.setContentsMargins(12, 12, 12, 12)
    layout.setVerticalSpacing(8)
    layout.setColumnStretch(1, 1)

    preset_heading = label(frame, 0, 0, preset_label, tooltip=preset_tooltip)
    layout.addWidget(preset_heading, 0, 0, 1, 2)
    layer_entry = entry(frame, 1, 1, ui_state, entry_var_name, tooltip=entry_tooltip)
    regex_label = label(frame, 2, 0, "Use Regex", tooltip=regex_tooltip)
    regex_switch = switch(frame, 2, 1, ui_state, regex_var_name)

    presets_list = list(presets.keys()) + ["custom"]

    def preset_set_layer_choice(selected: str):
        if not selected or selected not in presets_list:
            selected = presets_list[0]

        if selected == "custom":
            layer_entry.setVisible(True)
            layer_entry.setEnabled(True)
            regex_label.setVisible(True)
            regex_switch.setVisible(True)
        else:
            preset_def = presets.get(selected, [])
            if isinstance(preset_def, dict):
                patterns = preset_def.get("patterns", [])
                preset_uses_regex = bool(preset_def.get("regex", False))
            else:
                patterns = preset_def
                preset_uses_regex = False

            layer_entry.setEnabled(False)
            ui_state.get_var(entry_var_name).set(",".join(patterns))
            ui_state.get_var(regex_var_name).set(preset_uses_regex)

            regex_label.setVisible(False)
            regex_switch.setVisible(False)

            layer_entry.setVisible(selected != "full" or bool(patterns))

    ui_state.remove_all_var_traces(preset_var_name)

    layer_selector = options(
        frame, 1, 0, presets_list, ui_state, preset_var_name,
        command=preset_set_layer_choice,
    )
    layout.setAlignment(layer_selector, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

    ui_state.add_var_trace(preset_var_name, lambda: preset_set_layer_choice(
        ui_state.get_var(preset_var_name).get()
    ))

    preset_set_layer_choice(layer_selector.currentText())

    return frame


def icon_button(master: QWidget, row: int, column: int, text: str, command: Callable[[], None]) -> QPushButton:
    component = QPushButton(text, master)
    component.setFixedWidth(40)
    component.clicked.connect(command)
    _add(_layout(master), component, row, column, sticky="new")
    return component


class _ListActionButton(QPushButton):
    def paintEvent(self, event):
        option = QStyleOptionButton()
        self.initStyleOption(option)
        text = "x" if self.property("listAction") == "remove" else option.text
        option.text = ""
        painter = QStylePainter(self)
        painter.drawControl(QStyle.ControlElement.CE_PushButton, option)

        # Center the visible glyph, rather than the font's ascent/descent box.
        contents = QRectF(self.style().subElementRect(QStyle.SubElement.SE_PushButtonContents, option, self))
        font = self.font()
        font.setCapitalization(QFont.Capitalization.MixedCase)
        bounds = QFontMetricsF(font).tightBoundingRect(text)
        painter.setFont(font)
        painter.setPen(option.palette.color(QPalette.ColorRole.ButtonText))
        painter.drawText(contents.center() - bounds.center(), text)


def list_action_button(
        master: QWidget,
        action: Literal["remove", "clone"],
        command: Callable[[], None],
        *,
        row: int | None = None,
        column: int = 0,
        size: int = 20,
        tooltip: str | None = None,
) -> QPushButton:
    """Create a themed list action, optionally placed in a form grid."""
    component = _ListActionButton("x" if action == "remove" else "+", master)
    component.setProperty("listAction", action)
    component.setFixedSize(size, size)
    component.setAccessibleName("Remove" if action == "remove" else "Duplicate")
    if tooltip:
        _set_tooltip(component, tooltip)
    component.clicked.connect(command)
    if row is not None:
        _add(_layout(master), component, row, column, sticky="new", padx=0, pady=0)
    return component


def colored_icon_button(
        master: QWidget,
        row: int,
        column: int,
        text: str,
        fg_color,
        command: Callable[[], None],
        padx: int = 0,
) -> QPushButton:
    color = fg_color[0] if isinstance(fg_color, (tuple, list)) else fg_color
    component = QPushButton(text, master)
    component.setFixedSize(20, 20)
    component.setStyleSheet(f"QPushButton {{ background-color: {color}; border-radius: 2px; }}")
    component.clicked.connect(command)
    _add(_layout(master), component, row, column, sticky="new", padx=padx, pady=0)
    return component


def button(
        master: QWidget,
        row: int,
        column: int,
        text: str,
        command: Callable[[], None],
        tooltip: str | None = None,
        padx: int = PAD,
        pady: int = PAD,
        sticky: str = "new",
        width: int | None = None,
) -> QPushButton:
    component = QPushButton(text, master)
    component.clicked.connect(command)
    if width is not None:
        # ctk's width is a floor, not a cap: CTkButton never disables grid propagation,
        # so it grows past `width` to fit its label. Match that with setMinimumWidth.
        component.setMinimumWidth(width)
    if tooltip:
        _set_tooltip(component, tooltip)
    _add(_layout(master), component, row, column, sticky=sticky, padx=padx, pady=pady)
    return component


def preset_menu_button(
        master: QWidget,
        row: int,
        column: int,
        text: str,
        tree: list[tuple[str, Any]],
        command: Callable[[Any], None],
        sticky: str = "new",
) -> QToolButton:
    def build_menu(parent_menu: QMenu, nodes):
        for name, value in nodes:
            if isinstance(value, list):
                submenu = parent_menu.addMenu(name)
                build_menu(submenu, value)
            else:
                parent_menu.addAction(name, lambda v=value: command(v))

    menu = QMenu(master)
    build_menu(menu, tree)

    component = QToolButton(master)
    component.setText(text)
    component.setMenu(menu)
    component.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
    # the global QToolButton stylesheet (pyside6_util.create_application) enlarges the
    # menu-indicator, which switches this widget to the CSS box model and resets its
    # vertical padding away from the native style's — match a plain QPushButton's
    # height explicitly so it doesn't grow taller than its siblings.
    reference_button = QPushButton("", master)
    component.setFixedHeight(reference_button.sizeHint().height())
    reference_button.deleteLater()
    _add(_layout(master), component, row, column, sticky=sticky)
    return component


# ---------------------------------------------------------------------------
# Bound widgets
# ---------------------------------------------------------------------------

class NoScrollComboBox(QComboBox):
    # scrolling over a closed combo box is an easy way to accidentally change
    # its value while just scrolling the surrounding page; only scroll while open
    def wheelEvent(self, event: QWheelEvent):
        if self.view().isVisible():
            super().wheelEvent(event)
        else:
            event.ignore()


class SelectedTextComboBox(NoScrollComboBox):
    """Prefer the selected text's width, but leave room to shrink in narrow forms."""

    def __init__(self, parent):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        self.currentTextChanged.connect(self.updateGeometry)

    def _text_size(self, text: str) -> QSize:
        option = QStyleOptionComboBox()
        self.initStyleOption(option)
        metrics = self.fontMetrics()
        size = QSize(metrics.horizontalAdvance(text), metrics.height())
        return self.style().sizeFromContents(QStyle.ContentsType.CT_ComboBox, option, size, self)

    def sizeHint(self):
        return self._text_size(self.currentText())

    def minimumSizeHint(self):
        return self._text_size("…")


def options(
        master: QWidget,
        row: int,
        column: int,
        values: list[str],
        ui_state: BaseUIState,
        var_name: str,
        command: Callable[[str], None] | None = None,
        *,
        max_width: int | None = None,
        fit_to_selection: bool = False,
) -> QComboBox:
    """Create a dropdown with an optional character-based max_width."""
    var = ui_state.get_var(var_name)
    combo = SelectedTextComboBox(master) if fit_to_selection else NoScrollComboBox(master)
    combo.addItems(values)
    combo.setCurrentText(str(var.get()))

    _updating = False

    def on_combo(text: str):
        nonlocal _updating
        if _updating:
            return
        _updating = True
        var.set(text)
        _updating = False
        if command:
            command(text)

    def on_var(value):
        nonlocal _updating
        if _updating:
            return
        _updating = True
        combo.setCurrentText(str(value))
        _updating = False

    combo.currentTextChanged.connect(on_combo)
    cb_id = var._bind_widget(on_var)
    combo.destroyed.connect(lambda: var._unbind_widget(cb_id))
    _add(_layout(master), combo, row, column)
    _set_max_width(combo, max_width)
    return combo


def options_adv(
        master: QWidget,
        row: int,
        column: int,
        values: list[str],
        ui_state: BaseUIState,
        var_name: str,
        command: Callable[[str], None] | None = None,
        adv_command: Callable[[], None] | None = None,
        *,
        fit_to_selection: bool = False,
) -> tuple[QWidget, dict]:
    frame = QWidget(master)
    frame_lo = QGridLayout(frame)
    frame_lo.setContentsMargins(0, 0, 0, 0)
    frame_lo.setColumnStretch(0, 1)
    _add(_layout(master), frame, row, column, sticky="nw" if fit_to_selection else "new", padx=0, pady=0)

    combo = options(frame, 0, 0, values, ui_state, var_name, command=command,
                    fit_to_selection=fit_to_selection)

    adv_btn = QPushButton("…", frame)
    adv_btn.setFixedWidth(20)
    if adv_command:
        adv_btn.clicked.connect(adv_command)
    _add(frame_lo, adv_btn, 0, 1, sticky="nsew", padx=(0, PAD), pady=PAD)

    if command:
        command(ui_state.get_var(var_name).get())

    return frame, {'component': combo, 'button_component': adv_btn}


def options_kv(
        master: QWidget,
        row: int,
        column: int,
        values: list[tuple[str, Any]],
        ui_state: BaseUIState,
        var_name: str,
        command: Callable[[Any], None] | None = None,
        sticky: str = "new",
) -> QComboBox:
    var = ui_state.get_var(var_name)
    keys = [key for key, _ in values]
    str_values = [str(v) for _, v in values]

    if var.get() not in str_values and keys:
        # store the str repr — UIState's enum trace looks up var_type[string]
        var.set(str(values[0][1]))

    _updating = False

    def on_combo(key: str):
        nonlocal _updating
        if _updating:
            return
        _updating = True
        for k, v in values:
            if key == k:
                var.set(str(v))
                if command:
                    command(v)
                break
        _updating = False

    def on_var(value):
        nonlocal _updating
        if _updating:
            return
        _updating = True
        for k, v in values:
            if str(value) == str(v):
                combo.setCurrentText(k)
                if command:
                    command(v)
                break
        _updating = False

    combo = NoScrollComboBox(master)
    combo.addItems(keys)
    # set initial display from current var value
    for k, v in values:
        if str(var.get()) == str(v):
            combo.setCurrentText(k)
            break

    combo.currentTextChanged.connect(on_combo)
    cb_id = var._bind_widget(on_var)
    combo.destroyed.connect(lambda: var._unbind_widget(cb_id))
    _add(_layout(master), combo, row, column, sticky=sticky)

    # match CTK behavior: fire initial command with the current value
    if command:
        current = var.get()
        for _, v in values:
            if str(current) == str(v):
                command(v)
                break

    return combo


def switch(
        master: QWidget,
        row: int,
        column: int,
        ui_state: BaseUIState,
        var_name: str,
        command: Callable[[], None] | None = None,
        text: str = "",
        width: int | None = None,
) -> QCheckBox:
    var = ui_state.get_var(var_name)
    component = QCheckBox(text, master)
    component.setChecked(bool(var.get()))

    if command:
        trace_id = ui_state.add_var_trace(var_name, command)
        component.destroyed.connect(lambda: ui_state.remove_var_trace(var_name, trace_id))

    _updating = False

    def on_toggle(checked: bool):
        nonlocal _updating
        if _updating:
            return
        _updating = True
        var.set(checked)
        _updating = False

    def on_var(value):
        nonlocal _updating
        if _updating:
            return
        _updating = True
        component.setChecked(bool(value))
        _updating = False

    component.toggled.connect(on_toggle)
    cb_id = var._bind_widget(on_var)
    component.destroyed.connect(lambda: var._unbind_widget(cb_id))

    if width is not None:
        component.setFixedWidth(width)
    lo = _layout(master)
    lo.addWidget(component, row, column)
    lo.setAlignment(component, Qt.AlignVCenter | Qt.AlignLeft)
    return component


def progress(master: QWidget, row: int, column: int) -> QProgressBar:
    component = QProgressBar(master)
    component.setRange(0, 1000)
    component.setValue(0)
    component.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    component.setFixedHeight(12)
    _add(_layout(master), component, row, column, sticky="ew")
    return component


def double_progress(
        master: QWidget,
        row: int,
        column: int,
        label_1: str,
        label_2: str,
) -> tuple[Callable, Callable]:
    frame = QWidget(master)
    lo = QGridLayout(frame)
    lo.setContentsMargins(0, 0, 0, 0)
    lo.setColumnStretch(1, 1)

    label_1_component = QLabel(label_1, frame)
    label_2_component = QLabel(label_2, frame)
    progress_1_component = QProgressBar(frame)
    progress_2_component = QProgressBar(frame)
    description_1_component = QLabel("", frame)
    description_2_component = QLabel("", frame)

    for p in (progress_1_component, progress_2_component):
        p.setRange(0, 1000)
        p.setValue(0)
        p.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        p.setFixedHeight(16)

    lo.addWidget(label_1_component,       0, 0)
    lo.addWidget(progress_1_component,    0, 1)
    lo.addWidget(description_1_component, 0, 2)
    lo.addWidget(label_2_component,       1, 0)
    lo.addWidget(progress_2_component,    1, 1)
    lo.addWidget(description_2_component, 1, 2)

    _add(_layout(master), frame, row, column, sticky="nsew")

    def set_1(value: int | float, max_value: int | float):
        progress_1_component.setValue(int(value / max_value * 1000))
        description_1_component.setText(f"{value}/{max_value}")

    def set_2(value: int | float, max_value: int | float):
        progress_2_component.setValue(int(value / max_value * 1000))
        description_2_component.setText(f"{value}/{max_value}")

    return set_1, set_2


def section_frame(parent: QWidget, row: int, col: int = 0, colspan: int = 1) -> QFrame:
    frame = QFrame(parent)
    frame.setFrameShape(QFrame.Shape.StyledPanel)
    frame.setProperty("formPanel", True)
    _layout(parent).addWidget(frame, row, col, 1, colspan)
    frame_lo = _layout(frame)
    frame_lo.setColumnStretch(0, 1)
    frame_lo.setContentsMargins(PAD, PAD, PAD, PAD)
    return frame


def inline_frame(parent: QWidget, row: int, col: int, columnspan: int = 1) -> QWidget:
    frame = QWidget(parent)
    _layout(frame)
    _layout(parent).addWidget(frame, row, col, 1, columnspan)
    return frame


def form_section(parent: QWidget, row: int, col: int, title: str, *, colspan: int = 1) -> QWidget:
    """Create a titled panel and return its body for field construction."""
    container = section_frame(parent, row, col, colspan=colspan)
    _layout(parent).setAlignment(container, Qt.AlignmentFlag.AlignTop)
    header_label(container, 0, 0, title)
    body = inline_frame(container, 1, 0)
    _layout(body).setColumnStretch(1, 1)
    _layout(container).setRowStretch(2, 1)
    return body


def toggle_details(
        parent: QWidget,
        title: str,
        ui_state: BaseUIState,
        var_name: str,
        tooltip: str | None = None,
) -> tuple[QtQuickSwitch, QLabel, CollapsibleForm]:
    """Build a bound enable row; the caller controls when its details expand."""
    summary = inline_frame(parent, 0, 0, columnspan=2)
    field_label = label(summary, 0, 0, title, tooltip=tooltip)
    switch = QtQuickSwitch(summary, ui_state, var_name, title)
    _add(_layout(summary), switch, 0, 1, sticky="vw")
    if tooltip:
        _set_tooltip(summary, tooltip)
        _set_tooltip(switch, tooltip)
    field_label.setBuddy(switch)
    _pack_form(summary)
    _DisabledToolTipFilter(switch, summary, field_label)
    details = CollapsibleForm(parent)
    _layout(details.content).setColumnStretch(1, 1)
    _layout(parent).addWidget(details, 1, 0, 1, 2)
    return switch, field_label, details


# ---------------------------------------------------------------------------
# Pure helper (toolkit-neutral)
# ---------------------------------------------------------------------------

def set_widget_enabled(widget: QWidget, enabled: bool) -> None:
    widget.setEnabled(enabled)


def set_label_text(label: QLabel, text: str) -> None:
    label.setText(str(text))


def call_after(widget: QWidget, delay_ms: int, func) -> None:
    QTimer.singleShot(delay_ms, widget, func)
