import contextlib
import html
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from modules.util.enum.PathIOType import PathIOType
from modules.util.enum.TimeUnit import TimeUnit
from modules.util.path_util import supported_image_extensions, supported_video_extensions
from modules.util.ui.pyside6_validation import PySide6FieldValidator, PySide6PathValidator
from modules.util.ui.UIState import BaseUIState

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

PAD = 10


# ---------------------------------------------------------------------------
# PySide6-only helpers
# ---------------------------------------------------------------------------

def _layout(master: QWidget) -> QGridLayout:
    lo = master.layout()
    if lo is None:
        lo = QGridLayout(master)
        lo.setContentsMargins(0, 0, 0, 0)
        lo.setSpacing(PAD)
        master.setLayout(lo)
    return lo


def _set_tooltip(component: QWidget, text: str, wide: bool = False) -> None:
    # plain QToolTip text is rendered on a single line; wrap it as rich text
    # with a max-width so it matches Ctk's wraplength of 180/350px
    width = 350 if wide else 180
    component.setToolTip(f'<p style="max-width: {width}px;">{html.escape(text)}</p>')


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
) -> QLabel:
    component = QLabel(text, master)
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
    if underline:
        font = component.font()
        font.setUnderline(True)
        component.setFont(font)
    layout = _layout(master)
    layout.addWidget(component, row, column)
    layout.setAlignment(component, cell_alignment)
    return component


# ---------------------------------------------------------------------------
# Compound widgets
# ---------------------------------------------------------------------------

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
) -> QLineEdit:
    var = ui_state.get_var(var_name)

    component = QLineEdit(master)
    component.setMinimumWidth(width)
    _add(_layout(master), component, row, column, sticky=sticky)

    if command:
        trace_id = ui_state.add_var_trace(var_name, command)
        component.destroyed.connect(lambda: ui_state.remove_var_trace(var_name, trace_id))

    if tooltip:
        _set_tooltip(component, tooltip, wide_tooltip)

    if validator_factory is not None:
        validator = validator_factory(
            component, var, ui_state, var_name,
            extra_validate=extra_validate,
            required=required,
        )
    else:
        validator = PySide6FieldValidator(
            component, var, ui_state, var_name,
            extra_validate=extra_validate,
            required=required,
        )
    validator.attach()
    component._validator = validator  # type: ignore[attr-defined]

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
) -> QWidget:
    frame = QWidget(master)
    frame_lo = QGridLayout(frame)
    frame_lo.setContentsMargins(0, 0, 0, 0)
    frame_lo.setSpacing(0)
    frame_lo.setColumnStretch(0, 1)
    _add(_layout(master), frame, row, column, sticky="new", padx=0, pady=0, colspan=columnspan)

    def _path_validator_factory(comp, var, state, name, **kw):
        return PySide6PathValidator(comp, var, state, name, io_type=io_type, **kw)

    entry_component = entry(
        frame, 0, 0, ui_state, var_name,
        validator_factory=_path_validator_factory,
        extra_validate=extra_validate,
        required=required,
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
            if path_modifier:
                chosen = path_modifier(chosen)
            chosen_str = str(chosen)
            ui_state.get_var(var_name).set(chosen_str)
            if command:
                command(chosen_str)

    btn = QPushButton("...", frame)
    btn.setFixedWidth(40)
    btn.clicked.connect(_open_dialog)
    frame_lo.addWidget(btn, 0, 1)

    return frame


def time_entry(
        master: QWidget,
        row: int,
        column: int,
        ui_state: BaseUIState,
        var_name: str,
        unit_var_name: str,
        supports_time_units: bool = True,
) -> QWidget:
    frame = QWidget(master)
    _add(_layout(master), frame, row, column, sticky="new", padx=0, pady=0)

    entry(frame, 0, 0, ui_state, var_name, width=50)

    values = [str(x) for x in list(TimeUnit)]
    if not supports_time_units:
        values = [str(x) for x in list(TimeUnit) if not x.is_time_unit()]

    options(frame, 0, 1, values, ui_state, unit_var_name)

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
        frame_color=None,
) -> QWidget:
    frame = QWidget(master)
    _layout(master).addWidget(frame, row, column)

    label(frame, 0, 0, preset_label, tooltip=preset_tooltip)

    layer_entry = entry(frame, 1, 0, ui_state, entry_var_name, tooltip=entry_tooltip)
    _layout(frame).addWidget(layer_entry, 1, 0, 1, 2)  # span 2 columns

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
        frame, 0, 1, presets_list, ui_state, preset_var_name,
        command=preset_set_layer_choice,
    )

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


# ---------------------------------------------------------------------------
# Bound widgets
# ---------------------------------------------------------------------------

def options(
        master: QWidget,
        row: int,
        column: int,
        values: list[str],
        ui_state: BaseUIState,
        var_name: str,
        command: Callable[[str], None] | None = None,
) -> QComboBox:
    var = ui_state.get_var(var_name)
    combo = QComboBox(master)
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
) -> tuple[QWidget, dict]:
    frame = QWidget(master)
    frame_lo = QGridLayout(frame)
    frame_lo.setContentsMargins(0, 0, 0, 0)
    frame_lo.setColumnStretch(0, 1)
    _add(_layout(master), frame, row, column, sticky="new", padx=0, pady=0)

    combo = options(frame, 0, 0, values, ui_state, var_name, command=command)

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

    combo = QComboBox(master)
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


def section_frame(parent: QWidget, row: int, col: int = 0, colspan: int = 1) -> "QFrame":
    from PySide6.QtWidgets import QFrame
    frame = QFrame(parent)
    frame.setFrameShape(QFrame.Shape.StyledPanel)
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


# ---------------------------------------------------------------------------
# Pure helper (toolkit-neutral)
# ---------------------------------------------------------------------------

def set_widget_enabled(widget: QWidget, enabled: bool) -> None:
    widget.setEnabled(enabled)


def set_label_text(label: QLabel, text: str) -> None:
    label.setText(str(text))


def call_after(widget: QWidget, delay_ms: int, func) -> None:
    QTimer.singleShot(delay_ms, widget, func)
