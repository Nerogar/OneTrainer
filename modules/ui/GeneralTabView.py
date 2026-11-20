from functools import partial

from modules.ui.GeneralTabController import GeneralTabController
from modules.util.enum.GradientReducePrecision import GradientReducePrecision
from modules.util.enum.PathIOType import PathIOType
from modules.util.ui import pyside6_components
from modules.util.ui.pyside6_util import fit_form, get_theme_controller
from modules.util.ui.validation_helpers import validate_validation_timesteps

from PySide6.QtCore import QSignalBlocker, Qt, QTimer
from PySide6.QtWidgets import QWidget


class BaseGeneralTabView:
    def __init__(self, components):
        self.components = components

    def _build_main(self, frame, controller, ui_state, start_row: int = 0) -> int:
        field = partial(self.components.labeled_field, frame, ui_state)
        field(start_row, self.components.path_entry, "workspace_dir", "Workspace Directory",
              "The directory where all files of this training run are saved",
              mode="dir", command=controller.on_workspace_dir_change, io_type=PathIOType.OUTPUT)

        field(start_row + 1, self.components.path_entry, "cache_dir", "Cache Directory",
              "The directory where cached data is saved",
              mode="dir", io_type=PathIOType.OUTPUT)

        field(start_row + 2, self.components.switch, "continue_last_backup", "Resume from backup",
              "Automatically continues training from the last backup saved in <workspace>/backup")

        field(start_row + 3, self.components.switch, "only_cache", "Only Cache",
              "Only populate the cache, without any training")
        return start_row + 4

    def _build_debugging(self, frame, controller, ui_state, start_row: int = 0) -> int:
        field = partial(self.components.labeled_field, frame, ui_state)
        field(start_row, self.components.switch, "debug_mode", "Debug mode",
              "Save debug information during the training into the debug directory")

        field(start_row + 1, self.components.path_entry, "debug_dir", "Debug Directory",
              "The directory where debug data is saved",
              mode="dir", io_type=PathIOType.OUTPUT)
        return start_row + 2

    def _build_tensorboard(self, frame, controller, ui_state, start_row: int = 0) -> int:
        field = partial(self.components.labeled_field, frame, ui_state)
        field(start_row, self.components.switch, "tensorboard", "Tensorboard",
              "Starts the Tensorboard Web UI during training")

        field(start_row + 1, self.components.switch, "tensorboard_always_on", "Always-On Tensorboard",
              "Keep Tensorboard accessible even when not training. Useful for monitoring completed training sessions.",
              command=controller.on_always_on_tensorboard_toggle)

        field(start_row + 2, self.components.switch, "tensorboard_expose", "Expose Tensorboard",
              "Exposes Tensorboard Web UI to all network interfaces (makes it accessible from the network)")

        field(start_row + 3, self.components.entry, "tensorboard_port", "Tensorboard Port",
              "Port to use for Tensorboard link",
              max_width=6)
        return start_row + 4

    def _build_validation(self, frame, controller, ui_state):
        field = partial(self.components.labeled_field, frame, ui_state)
        field(0, self.components.time_entry, "validate_after", "Validate after",
              "The interval used when validate training",
              unit_var_name="validate_after_unit")

        validation_timesteps_tooltip = (
            "Comma-separated validation timesteps from 0 to 999 (for example, 100,499,900). "
            "Higher values test noisier images; lower values test finer details. "
            "Each distinct timestep gets a separate graph. VAE validation runs once without timesteps."
        )

        field(1, self.components.entry, "validation_timesteps", "Validation Timesteps",
              validation_timesteps_tooltip,
              required=True, extra_validate=validate_validation_timesteps, tooltip=validation_timesteps_tooltip)

        field(2, self.components.switch, "validation_report_per_concept", "Report per val concept",
              "Report validation timesteps in a separate Tensorboard accordion for each concept. "
              "Show the average validation loss and training loss in the loss group.")

        field(3, self.components.entry, "early_stopping_patience", "Early Stopping Patience",
              "Stop after X epochs without validation loss improvement. "
              "0 disables early stopping.",
              required=True, max_width=3)

        field(4, self.components.entry, "early_stopping_min_delta", "Minimum Loss Decrease",
              "Minimum absolute decrease required to reset patience. "
              "Leave blank to reset whenever validation loss reaches a new lowest value.")

    def _build_device(self, frame, controller, ui_state):
        field = partial(self.components.labeled_field, frame, ui_state)
        field(0, self.components.entry, "dataloader_threads", "Dataloader Threads",
              "Number of threads used for the data loader. Increase if your GPU has room during caching, decrease if it's going out of memory during caching.",
              required=True, max_width=3)

        field(1, self.components.entry, "train_device", "Train Device",
              "The device used for training. Can be \"cuda\", \"cuda:0\", \"cuda:1\" etc. Default:\"cuda\". Must be \"cuda\" for multi-GPU training.",
              required=True, max_width=7)

        field(2, self.components.entry, "temp_device", "Temp Device",
              "The device used to temporarily offload models while they are not used. Default:\"cpu\"",
              max_width=7)

        field(3, self.components.switch, "async_offloading", "Async Offloading",
              "Overlaps CPU<->GPU transfers with computation using CUDA streams. Applies to every offloaded component")

    def _build_multi_gpu(self, frame, controller, ui_state):
        field = partial(self.components.labeled_field, frame, ui_state)
        field(0, self.components.entry, "device_indexes", "Device Indexes",
              "Multi-GPU: A comma-separated list of device indexes. If empty, all your GPUs are used. With a list such as \"0,1,3,4\" you can omit a GPU, for example an on-board graphics GPU.")

        field(1, self.components.options, "gradient_reduce_precision", "Gradient Reduce Precision",
              "WEIGHT_DTYPE: Reduce gradients between GPUs in your weight data type; can be imprecise, but more efficient than float32\n"
              "WEIGHT_DTYPE_STOCHASTIC: Sum up the gradients in your weight data type, but average them in float32 and stochastically round if your weight data type is bfloat16\n"
              "FLOAT_32: Reduce gradients in float32\n"
              "FLOAT_32_STOCHASTIC: Reduce gradients in float32; use stochastic rounding to bfloat16 if your weight data type is bfloat16",
              values=[str(x) for x in GradientReducePrecision])

        field(2, self.components.switch, "fused_gradient_reduce", "Fused Gradient Reduce",
              "Multi-GPU: Gradient synchronisation during the backward pass. Can be more efficient, especially with Async Gradient Reduce")

        field(3, self.components.switch, "async_gradient_reduce", "Async Gradient Reduce",
              "Multi-GPU: Asynchroniously start the gradient reduce operations during the backward pass. Can be more efficient, but requires some VRAM.")

        field(4, self.components.entry, "async_gradient_reduce_buffer", "Buffer size (MB)",
              "Multi-GPU: Maximum VRAM for \"Async Gradient Reduce\", in megabytes. A multiple of this value can be needed if combined with \"Fused Back Pass\" and/or \"Layer offload fraction\"")

    def _build_input_validation(self, frame, controller, ui_state):
        field = partial(self.components.labeled_field, frame, ui_state)
        field(0, self.components.switch, "auto_correct_input", "Clean Up Input",
              "On leaving a field, pressing Enter, or starting training: trims surrounding whitespace, "
              "removes matching outer path quotes, expands ~ in local paths, and converts Windows separators. "
              "Invalid input is flagged for you to fix.")

        field(1, self.components.switch, "prevent_overwrites", "Prevent Overwrites",
              "When enabled, output paths that already exist on disk will be flagged as invalid to avoid accidental overwrites")


class GeneralTabView(BaseGeneralTabView, QWidget):
    def __init__(self, parent, controller: GeneralTabController, ui_state):
        QWidget.__init__(self, parent)
        BaseGeneralTabView.__init__(self, pyside6_components)
        self.controller = controller
        self.ui_state = ui_state
        self._refreshing = False
        self._loading = False
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.timeout.connect(self.refresh_availability)

        layout = self.components._layout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)
        left = self.components.section_frame(self, 0, 0)
        right = self.components.inline_frame(self, 0, 1)
        self.components._layout(right).setColumnStretch(0, 1)
        self.components._layout(right).setColumnStretch(1, 1)
        self.validation_column = self.components.inline_frame(right, 0, 0)
        self.device_column = self.components.inline_frame(right, 0, 1)
        for column in (self.validation_column, self.device_column):
            column_layout = self.components._layout(column)
            column_layout.setColumnStretch(0, 1)
            column_layout.setRowStretch(2, 1)
        # One grid shares the label column across all sections in this panel.
        # Its natural width follows the widest label without fixed pixel widths.
        left_layout = self.components._layout(left)
        left_layout.setColumnStretch(0, 0)
        left_layout.setColumnStretch(1, 1)
        row = 0
        for title, builder in (
            ("Main", self._build_main),
            ("Debugging", self._build_debugging),
            ("Tensorboard", self._build_tensorboard),
        ):
            self.components.header_label(left, row, 0, title, top_spacing=8 if row else 0, colspan=2)
            row = builder(left, controller, ui_state, start_row=row + 1)

        validation = self.components.form_section(self.validation_column, 0, 0, "Validation Loss")
        self.validation_panel = validation.parentWidget()
        self.validation_switch, _, self.validation_details = self.components.toggle_details(
            validation, "Enable Validation", ui_state, "validation",
            tooltip="Enable validation steps and display val loss in Tensorboard (if TB is enabled)",
        )
        self._build_validation(self.validation_details.content, controller, ui_state)
        fit_form(self.validation_details.content)
        self._build_device(self.components.form_section(self.device_column, 0, 0, "Device Settings"), controller, ui_state)
        appearance = self.components.form_section(self.validation_column, 1, 0, "Appearance")
        self.appearance_panel = appearance.parentWidget()
        self._build_appearance(appearance, controller, ui_state)
        self.validation_details.transition_finished.connect(self._align_top_panels)

        multi = self.components.form_section(right, 1, 0, "Multi-GPU", colspan=2)
        self.multi_gpu_switch, self.multi_gpu_label, self.multi_gpu_details = self.components.toggle_details(
            multi, "Enable Multi-GPU", ui_state, "multi_gpu",
        )
        self.multi_gpu_summary = self.multi_gpu_switch.parentWidget()
        self._build_multi_gpu(self.multi_gpu_details.content, controller, ui_state)
        fit_form(self.multi_gpu_details.content)
        self._build_input_validation(
            self.components.form_section(right, 2, 0, "Input Validation", colspan=2), controller, ui_state,
        )
        self.components._pack_form(left)
        self.components._pack_form(right)
        for details in (self.validation_details, self.multi_gpu_details):
            # Reserve the form width while collapsed so only height animates.
            details.parentWidget().setMinimumWidth(details.minimumSizeHint().width())

        # Nested states own their own trace registries. Defer bulk-load callbacks
        # until both cloud.enabled and multi_gpu have their final values.
        for state, name in ((ui_state, "multi_gpu"), (ui_state, "validation"),
                            (ui_state.get_var("cloud"), "enabled")):
            trace_id = state.add_var_trace(name, self.refresh_availability)
            self.destroyed.connect(lambda _obj=None, state=state, name=name, trace_id=trace_id:
                                   state.remove_var_trace(name, trace_id))
        self.refresh_availability()

    def _position_appearance(self, validation_enabled):
        column = self.device_column if validation_enabled else self.validation_column
        previous = self.appearance_panel.parentWidget()
        if previous is column:
            return
        previous.layout().removeWidget(self.appearance_panel)
        column.layout().addWidget(self.appearance_panel, 1, 0, Qt.AlignmentFlag.AlignTop)
        self.appearance_panel.show()

    def _align_top_panels(self):
        # Animate at natural height. Once settled, fill the upper row: the
        # collapsed stack shares spare height; expanded Appearance sits below Device.
        settled = not self.validation_details.is_animating
        expanded = settled and self.validation_details.expanded
        collapsed = settled and not self.validation_details.expanded
        validation_layout = self.validation_column.layout()
        device_layout = self.device_column.layout()
        validation_layout.setRowStretch(0, 1 if settled else 0)
        validation_layout.setRowStretch(1, 1 if collapsed else 0)
        validation_layout.setRowStretch(2, 0 if settled else 1)
        validation_layout.setAlignment(
            self.validation_panel, Qt.AlignmentFlag(0) if settled else Qt.AlignmentFlag.AlignTop,
        )
        device_layout.setRowStretch(1, 1 if expanded else 0)
        device_layout.setRowStretch(2, 0 if expanded else 1)
        self.appearance_panel.parentWidget().layout().setAlignment(
            self.appearance_panel,
            Qt.AlignmentFlag.AlignBottom if expanded else Qt.AlignmentFlag(0) if collapsed else Qt.AlignmentFlag.AlignTop,
        )

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
            multi_gpu_enabled = bool(self.ui_state.get_var("multi_gpu").get())
            available, tooltip = self.controller.get_multi_gpu_availability(cloud)
            if not available and multi_gpu_enabled:
                tooltip = f"{tooltip} Turn off Multi-GPU or enable Cloud before starting local training."
            can_toggle = available or multi_gpu_enabled
            self.multi_gpu_switch.setEnabled(can_toggle)
            self.multi_gpu_label.setEnabled(can_toggle)
            for widget in (self.multi_gpu_summary, self.multi_gpu_label, self.multi_gpu_switch):
                self.components._set_tooltip(widget, tooltip)
            self.multi_gpu_details.set_expanded(available and multi_gpu_enabled, animate=animate)
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
        layout = self.components._layout(frame)
        layout.setHorizontalSpacing(16)
        self.components._add(layout, self.theme_dropdown, 0, 1, sticky="vw")
        fit_form(frame)

    def _update_theme_dropdown(self, preference):
        with QSignalBlocker(self.theme_dropdown):
            self.theme_dropdown.setCurrentIndex(self.theme_dropdown.findData(preference))
