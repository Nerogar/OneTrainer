from modules.util.enum.GradientReducePrecision import GradientReducePrecision
from modules.util.enum.PathIOType import PathIOType
from modules.util.ui import pyside6_components
from modules.util.ui.pyside6_util import get_theme_controller
from modules.util.ui.QtQuickSwitch import QtQuickSwitch
from modules.util.ui.validation_helpers import validate_validation_timesteps

from PySide6.QtCore import (
    QAbstractAnimation,
    QEasingCurve,
    QEvent,
    QSignalBlocker,
    QSize,
    Qt,
    QTimer,
    QVariantAnimation,
)
from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QApplication, QGridLayout, QLabel, QSizePolicy, QToolTip, QWidget


class _AnimatedDetails(QWidget):
    """Clip an intact form while its height changes; keep all bindings alive."""

    def __init__(self, parent):
        super().__init__(parent)
        self.content = QWidget(self)
        self.content.setEnabled(False)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.expanded = False
        self.animation = QVariantAnimation(self)
        self.animation.setDuration(180)
        self.animation.setEasingCurve(QEasingCurve.Type.InOutCubic)
        self.animation.valueChanged.connect(self.setFixedHeight)
        self.animation.finished.connect(self._finish)
        self.setFixedHeight(0)
        self.hide()

    def sizeHint(self):
        return self.content.sizeHint()

    def minimumSizeHint(self):
        return QSize(self.content.minimumSizeHint().width(), 0)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # The content retains its natural height and is clipped by this widget.
        self.content.setGeometry(0, 0, self.width(), self.content.sizeHint().height())

    def set_expanded(self, expanded, *, animate):
        if expanded == self.expanded:
            if not animate:
                self.animation.stop()
                self._finish()
            return
        self.expanded = expanded
        height = self.height() if not self.isHidden() else 0
        self.animation.stop()
        self.content.setEnabled(expanded)
        if not animate:
            self._finish()
            return
        self.show()
        self.animation.setStartValue(height)
        self.animation.setEndValue(self.content.sizeHint().height() if expanded else 0)
        self.animation.start()

    def _finish(self):
        if self.expanded:
            self.setMinimumHeight(0)
            self.setMaximumHeight(16777215)
            self.show()
        else:
            self.setFixedHeight(0)
            self.hide()
        self.updateGeometry()


class GeneralTabView(QWidget):
    def __init__(self, parent, controller, ui_state):
        super().__init__(parent)
        self.components = pyside6_components
        self.controller = controller
        self.ui_state = ui_state
        self._refreshing = False
        self._loading = False
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.timeout.connect(self.refresh_availability)

        layout = self.components._layout(self)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)
        left = self.components.section_frame(self, 0, 0)
        left.setProperty("generalPanel", True)
        right = self.components.inline_frame(self, 0, 1)
        self.components._layout(right).setColumnStretch(0, 1)
        self.components._layout(right).setColumnStretch(1, 1)
        self.validation_column = self.components.inline_frame(right, 0, 0)
        self.device_column = self.components.inline_frame(right, 0, 1)
        for column in (self.validation_column, self.device_column):
            column_layout = self.components._layout(column)
            column_layout.setColumnStretch(0, 1)
            column_layout.setRowStretch(2, 1)
        for row, (title, builder) in enumerate((
            ("Main", self._build_main),
            ("Debugging", self._build_debugging),
            ("Tensorboard", self._build_tensorboard),
        )):
            group = self._group(left, row, 0, title, panel=False)
            builder(group, controller, ui_state)

        validation = self._group(self.validation_column, 0, 0, "Validation Loss")
        self.validation_panel = validation.parentWidget()
        self.validation_switch, _, self.validation_details = self._toggle_details(
            validation, "Enable Validation", "validation",
        )
        self._build_validation(self.validation_details.content, controller, ui_state)
        self._build_device(self._group(self.device_column, 0, 0, "Device Settings"), controller, ui_state)
        appearance = self._group(self.validation_column, 1, 0, "Appearance")
        self.appearance_panel = appearance.parentWidget()
        self._build_appearance(appearance, controller, ui_state)
        self.validation_details.animation.finished.connect(self._align_top_panels)

        multi = self._group(right, 1, 0, "Multi-GPU", colspan=2)
        self.multi_gpu_switch, self.multi_gpu_label, self.multi_gpu_details = self._toggle_details(
            multi, "Enable Multi-GPU", "multi_gpu",
        )
        self.multi_gpu_summary = self.multi_gpu_switch.parentWidget()
        for widget in (self.multi_gpu_summary, self.multi_gpu_label, self.multi_gpu_switch):
            widget.installEventFilter(self)
        self._build_multi_gpu(self.multi_gpu_details.content, controller, ui_state)
        self._build_input_validation(
            self._group(right, 2, 0, "Input Validation", colspan=2), controller, ui_state,
        )
        self.components._pack_form(left)
        self.components._pack_form(right)
        self._style_labels_and_rows()
        for details in (self.validation_details, self.multi_gpu_details):
            # Reserve the form width while collapsed so only height animates.
            details.parentWidget().setMinimumWidth(details.minimumSizeHint().width())
        self._apply_panel_colors()
        QApplication.instance().paletteChanged.connect(self._apply_panel_colors)

        # Nested states own their own trace registries. Defer bulk-load callbacks
        # until both cloud.enabled and multi_gpu have their final values.
        for state, name in ((ui_state, "multi_gpu"), (ui_state, "validation"),
                            (ui_state.get_var("cloud"), "enabled")):
            trace_id = state.add_var_trace(name, self.refresh_availability)
            self.destroyed.connect(lambda _obj=None, state=state, name=name, trace_id=trace_id:
                                   state.remove_var_trace(name, trace_id))
        self.refresh_availability()

    def _group(self, parent, row, col, title, *, panel=True, colspan=1):
        container = (self.components.section_frame(parent, row, col, colspan=colspan) if panel
                     else self.components.inline_frame(parent, row, col, columnspan=colspan))
        if panel:
            container.setProperty("generalPanel", True)
            self.components._layout(parent).setAlignment(container, Qt.AlignmentFlag.AlignTop)
        heading = self.components.label(container, 0, 0, title)
        heading.setProperty("generalHeading", True)
        if title in ("Debugging", "Tensorboard"):
            heading.setContentsMargins(0, 8, 0, 0)
        body = self.components.inline_frame(container, 1, 0)
        self.components._layout(body).setColumnStretch(1, 1)
        self.components._layout(container).setRowStretch(2, 1)
        return body

    def _toggle_details(self, parent, title, name):
        summary = self.components.inline_frame(parent, 0, 0, columnspan=2)
        label = self.components.label(summary, 0, 0, title)
        switch = QtQuickSwitch(summary, self.ui_state, name, title)
        self.components._add(self.components._layout(summary), switch, 0, 1, sticky="vw")
        label.setBuddy(switch)
        self.components._pack_form(summary)
        details = _AnimatedDetails(parent)
        self.components._layout(details.content).setColumnStretch(1, 1)
        self.components._layout(parent).addWidget(details, 1, 0, 1, 2)
        return switch, label, details

    def _style_labels_and_rows(self):
        # Retain the platform font family; give headings a clear hierarchy.
        size = max(10.0, self.font().pointSizeF())
        for label in self.findChildren(QLabel):
            heading = bool(label.property("generalHeading"))
            label.setProperty("generalField", not heading)
            font = label.font()
            font.setPointSizeF(size + 1 if heading else size)
            font.setWeight(QFont.Weight.Bold if heading else QFont.Weight.Normal)
            label.setFont(font)
        for layout in self.findChildren(QGridLayout):
            layout.setVerticalSpacing(self.components.PAD + 4)

    def _position_appearance(self, validation_enabled):
        column = self.device_column if validation_enabled else self.validation_column
        previous = self.appearance_panel.parentWidget()
        if previous is column:
            return
        previous.layout().removeWidget(self.appearance_panel)
        column.layout().addWidget(self.appearance_panel, 1, 0, Qt.AlignmentFlag.AlignTop)
        self.appearance_panel.show()

    def _align_top_panels(self):
        # Let the reveal animate at its natural height, then share the upper
        # row's height: extra room goes below validation or above Appearance.
        aligned = (self.validation_details.expanded
                   and self.validation_details.animation.state() != QAbstractAnimation.State.Running)
        validation_layout = self.validation_column.layout()
        device_layout = self.device_column.layout()
        validation_layout.setRowStretch(0, 1 if aligned else 0)
        validation_layout.setRowStretch(2, 0 if aligned else 1)
        validation_layout.setAlignment(
            self.validation_panel, Qt.AlignmentFlag(0) if aligned else Qt.AlignmentFlag.AlignTop,
        )
        device_layout.setRowStretch(1, 1 if aligned else 0)
        device_layout.setRowStretch(2, 0 if aligned else 1)
        self.appearance_panel.parentWidget().layout().setAlignment(
            self.appearance_panel, Qt.AlignmentFlag.AlignBottom if aligned else Qt.AlignmentFlag.AlignTop,
        )

    def _apply_panel_colors(self, *_args):
        palette = QApplication.palette()
        base = palette.color(QPalette.ColorRole.Window)
        text = palette.color(QPalette.ColorRole.WindowText)
        dark = base.lightnessF() < 0.5

        def blend(first, second, amount):
            return QColor(*(round(a + (b - a) * amount) for a, b in zip(
                first.getRgb()[:3], second.getRgb()[:3], strict=True,
            ))).name()

        surface = blend(base, QColor("white"), 0.065 if dark else 0.65)
        border = blend(base, QColor("white" if dark else "black"), 0.13 if dark else 0.12)
        field = blend(text, base, 0.12)
        for switch in (self.validation_switch, self.multi_gpu_switch):
            switch.setClearColor(QColor(surface))
        self.setStyleSheet(f"""
            QFrame[generalPanel="true"] {{
                background-color: {surface};
                border: 1px solid {border};
                border-radius: 6px;
            }}
            QLabel[generalHeading="true"] {{ color: {text.name()}; }}
            QLabel[generalField="true"]:enabled {{ color: {field}; }}
        """)

    def eventFilter(self, watched, event):
        # Qt's disabled controls cannot be clicked, but must still explain why.
        if event.type() == QEvent.Type.ToolTip and not self.multi_gpu_switch.isEnabled():
            QToolTip.showText(event.globalPos(), watched.toolTip(), watched)
            return True
        return super().eventFilter(watched, event)

    def refresh_availability(self):
        if self.ui_state.updating:
            self._loading = True
            self._refresh_timer.start(0)
            return
        if self._refreshing:
            return
        self._refresh_timer.stop()
        self._refreshing = True
        try:
            animate = self.isVisible() and not self._loading
            cloud = bool(self.ui_state.get_var("cloud.enabled").get())
            count = self.controller._local_gpu_count
            available = cloud or (count is not None and count >= 2)
            self.multi_gpu_switch.setEnabled(available)
            self.multi_gpu_label.setEnabled(available)
            tooltip = ("Enable multi-GPU training" if available else
                       "Multi-GPU is unavailable because GPU detection failed." if count is None else
                       f"Multi-GPU requires at least two visible GPUs; {count} detected on this computer.")
            for widget in (self.multi_gpu_summary, self.multi_gpu_label, self.multi_gpu_switch):
                widget.setToolTip(tooltip)
            if not available and self.ui_state.get_var("multi_gpu").get():
                self.ui_state.get_var("multi_gpu").set(False)
            self.multi_gpu_details.set_expanded(
                available and bool(self.ui_state.get_var("multi_gpu").get()), animate=animate,
            )
            validation_enabled = bool(self.ui_state.get_var("validation").get())
            self._position_appearance(validation_enabled)
            self.validation_details.set_expanded(validation_enabled, animate=animate)
            self._align_top_panels()
        finally:
            self._refreshing = False
            self._loading = False

    def _build_appearance(self, frame, controller, ui_state):
        self.components.label(frame, 0, 0, 'Theme', tooltip='Application appearance, saved independently of training presets')
        self.theme_dropdown = self.components.NoScrollComboBox(frame)
        for label, preference in (('Dark', 'dark'), ('Light', 'light'), ('Match system', 'system')):
            self.theme_dropdown.addItem(label, preference)
        theme_controller = get_theme_controller()
        self._update_theme_dropdown(theme_controller.preference)
        self.theme_dropdown.currentIndexChanged.connect(
            lambda _: theme_controller.set_preference(self.theme_dropdown.currentData())
        )
        theme_controller.preference_changed.connect(self._update_theme_dropdown)
        self.components._add(self.components._layout(frame), self.theme_dropdown, 0, 1)

    def _update_theme_dropdown(self, preference):
        with QSignalBlocker(self.theme_dropdown):
            self.theme_dropdown.setCurrentIndex(self.theme_dropdown.findData(preference))

    def _build_main(self, frame, controller, ui_state):
        self.components.label(frame, 0, 0, "Workspace Directory",
                         tooltip="The directory where all files of this training run are saved")

        self.components.path_entry(frame, 0, 1, ui_state, "workspace_dir", mode="dir", command=controller._on_workspace_dir_change, io_type=PathIOType.OUTPUT)

        self.components.label(frame, 1, 0, "Cache Directory",
                         tooltip="The directory where cached data is saved")

        self.components.path_entry(frame, 1, 1, ui_state, "cache_dir", mode="dir", io_type=PathIOType.OUTPUT)

        self.components.label(frame, 2, 0, "Continue from last backup",
                         tooltip="Automatically continues training from the last backup saved in <workspace>/backup")

        self.components.switch(frame, 2, 1, ui_state, "continue_last_backup")

        self.components.label(frame, 3, 0, "Only Cache",
                         tooltip="Only populate the cache, without any training")

        self.components.switch(frame, 3, 1, ui_state, "only_cache")

    def _build_debugging(self, frame, controller, ui_state):
        self.components.label(frame, 0, 0, "Debug mode",
                         tooltip="Save debug information during the training into the debug directory")

        self.components.switch(frame, 0, 1, ui_state, "debug_mode")

        self.components.label(frame, 1, 0, "Debug Directory",
                         tooltip="The directory where debug data is saved")

        self.components.path_entry(frame, 1, 1, ui_state, "debug_dir", mode="dir", io_type=PathIOType.OUTPUT)

    def _build_tensorboard(self, frame, controller, ui_state):
        self.components.label(frame, 0, 0, "Tensorboard",
                         tooltip="Starts the Tensorboard Web UI during training")

        self.components.switch(frame, 0, 1, ui_state, "tensorboard")

        self.components.label(frame, 1, 0, "Always-On Tensorboard",
                         tooltip="Keep Tensorboard accessible even when not training. Useful for monitoring completed training sessions.")

        self.components.switch(frame, 1, 1, ui_state, "tensorboard_always_on", command=controller._on_always_on_tensorboard_toggle)

        self.components.label(frame, 2, 0, "Expose Tensorboard",
                         tooltip="Exposes Tensorboard Web UI to all network interfaces (makes it accessible from the network)")

        self.components.switch(frame, 2, 1, ui_state, "tensorboard_expose")

        self.components.label(frame, 3, 0, "Tensorboard Port",
                         tooltip="Port to use for Tensorboard link")

        self.components.entry(frame, 3, 1, ui_state, "tensorboard_port")

    def _build_validation(self, frame, controller, ui_state):
        self.components.label(frame, 0, 0, "Validate after",
                         tooltip="The interval used when validate training")

        self.components.time_entry(frame, 0, 1, ui_state, "validate_after", "validate_after_unit")

        validation_timesteps_tooltip = (
            "Comma-separated validation timesteps from 0 to 999 (for example, 100,499,900). "
            "Higher values test noisier images; lower values test finer details. "
            "Each distinct timestep gets a separate graph. VAE validation runs once without timesteps."
        )

        self.components.label(frame, 1, 0, "Validation Timesteps", tooltip=validation_timesteps_tooltip)

        self.components.entry(frame, 1, 1, ui_state, "validation_timesteps", required=True,
                              extra_validate=validate_validation_timesteps, tooltip=validation_timesteps_tooltip)

        self.components.label(frame, 2, 0, "Report per val concept",
                              tooltip="Report validation timesteps in a separate Tensorboard accordion for each concept. "
                                      "Show the average validation loss and training loss in the loss group.")

        self.components.switch(frame, 2, 1, ui_state, "validation_report_per_concept")

        self.components.label(frame, 3, 0, "Early Stopping Patience",
                              tooltip="Stop after X epochs without validation loss improvement. "
                                      "0 disables early stopping.")

        self.components.entry(frame, 3, 1, ui_state, "early_stopping_patience", required=True)

        self.components.label(frame, 4, 0, "Minimum Loss Decrease",
                              tooltip="Minimum absolute decrease required to reset patience. "
                                      "Leave blank to reset whenever validation loss reaches a new lowest value.")

        self.components.entry(frame, 4, 1, ui_state, "early_stopping_min_delta")

    def _build_device(self, frame, controller, ui_state):
        self.components.label(frame, 0, 0, "Dataloader Threads",
                         tooltip="Number of threads used for the data loader. Increase if your GPU has room during caching, decrease if it's going out of memory during caching.")

        self.components.entry(frame, 0, 1, ui_state, "dataloader_threads", required=True)

        self.components.label(frame, 1, 0, "Train Device",
                         tooltip="The device used for training. Can be \"cuda\", \"cuda:0\", \"cuda:1\" etc. Default:\"cuda\". Must be \"cuda\" for multi-GPU training.")

        self.components.entry(frame, 1, 1, ui_state, "train_device", required=True)

        self.components.label(frame, 3, 0, "Async Offloading",
                         tooltip="Overlaps CPU<->GPU transfers with computation using CUDA streams. Applies to every offloaded component")

        self.components.switch(frame, 3, 1, ui_state, "async_offloading")

        self.components.label(frame, 2, 0, "Temp Device",
                         tooltip="The device used to temporarily offload models while they are not used. Default:\"cpu\"")

        self.components.entry(frame, 2, 1, ui_state, "temp_device")

    def _build_multi_gpu(self, frame, controller, ui_state):
        self.components.label(frame, 0, 0, "Device Indexes",
                         tooltip="Multi-GPU: A comma-separated list of device indexes. If empty, all your GPUs are used. With a list such as \"0,1,3,4\" you can omit a GPU, for example an on-board graphics GPU.")

        self.components.entry(frame, 0, 1, ui_state, "device_indexes")

        self.components.label(frame, 1, 0, "Gradient Reduce Precision",
                         tooltip="WEIGHT_DTYPE: Reduce gradients between GPUs in your weight data type; can be imprecise, but more efficient than float32\n"
                                 "WEIGHT_DTYPE_STOCHASTIC: Sum up the gradients in your weight data type, but average them in float32 and stochastically round if your weight data type is bfloat16\n"
                                 "FLOAT_32: Reduce gradients in float32\n"
                                 "FLOAT_32_STOCHASTIC: Reduce gradients in float32; use stochastic rounding to bfloat16 if your weight data type is bfloat16",
                         wide_tooltip=True)

        self.components.options(frame, 1, 1, [str(x) for x in list(GradientReducePrecision)], ui_state,
                           "gradient_reduce_precision")

        self.components.label(frame, 2, 0, "Fused Gradient Reduce",
                         tooltip="Multi-GPU: Gradient synchronisation during the backward pass. Can be more efficient, especially with Async Gradient Reduce")

        self.components.switch(frame, 2, 1, ui_state, "fused_gradient_reduce")

        self.components.label(frame, 3, 0, "Async Gradient Reduce",
                         tooltip="Multi-GPU: Asynchroniously start the gradient reduce operations during the backward pass. Can be more efficient, but requires some VRAM.")

        self.components.switch(frame, 3, 1, ui_state, "async_gradient_reduce")

        self.components.label(frame, 4, 0, "Buffer size (MB)",
                         tooltip="Multi-GPU: Maximum VRAM for \"Async Gradient Reduce\", in megabytes. A multiple of this value can be needed if combined with \"Fused Back Pass\" and/or \"Layer offload fraction\"")

        self.components.entry(frame, 4, 1, ui_state, "async_gradient_reduce_buffer")

    def _build_input_validation(self, frame, controller, ui_state):
        self.components.label(frame, 1, 0, "Prevent Overwrites",
                         tooltip="When enabled, output paths that already exist on disk will be flagged as invalid to avoid accidental overwrites")

        self.components.switch(frame, 1, 1, ui_state, "prevent_overwrites")

        self.components.label(frame, 0, 0, "Clean Up Input",
                              tooltip="On leaving a field, pressing Enter, or starting training: trims surrounding whitespace, "
                                      "removes matching outer path quotes, expands ~ in local paths, and converts Windows separators. "
                                      "Invalid input is flagged for you to fix.")

        self.components.switch(frame, 0, 1, ui_state, "auto_correct_input")
