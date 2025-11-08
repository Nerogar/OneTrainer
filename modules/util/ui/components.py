import contextlib
import tkinter as tk
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog
from typing import Any, Literal

from modules.util.enum.TimeUnit import TimeUnit
from modules.util.path_util import supported_image_extensions
from modules.util.ui.ToolTip import ToolTip
from modules.util.ui.ui_utils import _register_drop_target
from modules.util.ui.UIState import UIState
from modules.util.ui.validation import ValidationResult, validate_basic_type, validate_file_path

import customtkinter as ctk
from customtkinter.windows.widgets.scaling import CTkScalingBaseClass
from PIL import Image

PAD = 10

@dataclass(frozen=True)
class ComponentValidationSettings:
    debounce_stop_typing_ms: int = 1700
    debounced_invalid_revert_ms: int = 1000
    focusout_invalid_revert_ms: int = 1200


COMPONENT_VALIDATION_SETTINGS = ComponentValidationSettings()

@dataclass
class ValidationState:
    """Tracks validation state for UI components."""
    status: Literal['error', 'warning'] | None = None
    message: str = ""

    def clear(self):
        self.status = None
        self.message = ""

    def set_error(self, message: str):
        self.status = 'error'
        self.message = message

    def set_warning(self, message: str):
        self.status = 'warning'
        self.message = message


class EntryValidationHandler:
    """Handles UI coordination for entry widget validation with debouncing and visual feedback."""

    def __init__(
        self,
        component: ctk.CTkEntry,
        var: tk.Variable,
        var_name: str,
        ui_state: UIState,
        custom_validator: Callable[[str], ValidationResult] | None = None,
        validation_state: ValidationState | None = None,
    ):
        self.component = component
        self.var = var
        self.var_name = var_name
        self.ui_state = ui_state
        self.custom_validator = custom_validator
        self.validation_state = validation_state  # Optional external validation state

        # Get border colors
        try:
            self.original_border_color = component.cget("border_color")
        except Exception:
            self.original_border_color = "gray50"

        self.error_border_color = "#dc3545"
        self.warning_border_color = "#ff9500"

        # Validation state
        self.validation_after_id = None
        self.revert_after_id = None
        self.touched = False
        self.last_valid_value = var.get()

        # Create tooltip (hover_only=False to prevent default hover behavior, wide=True for validation messages)
        self.validation_tooltip = ToolTip(component, text="", hover_only=False, track_movement=True, wide=True)
        component._validation_tooltip = self.validation_tooltip

    def should_show_tooltip(self) -> bool:
        """Check if validation tooltips are enabled in settings."""
        try:
            show_tooltips_var = self.ui_state.get_var("validation_show_tooltips")
            return bool(show_tooltips_var.get())
        except Exception:
            return True

    def validate_value(self, value: str, revert_delay_ms: int | None) -> bool:
        """
        Validate the current value and update UI accordingly.

        Returns:
            True if validation passed, False otherwise.
        """

        # Cancel any pending revert
        if self.revert_after_id:
            with contextlib.suppress(Exception):
                self.component.after_cancel(self.revert_after_id)
            self.revert_after_id = None

        meta = self.ui_state.get_field_metadata(self.var_name)

        # Perform basic type validation
        basic_result = validate_basic_type(
            value=value,
            declared_type=meta.type,
            nullable=meta.nullable,
            default_val=meta.default
        )

        # If basic validation fails, handle failure
        if not basic_result.ok:
            return self._fail(basic_result.message, revert_delay_ms)

        # Custom validation
        if self.custom_validator:
            result = self.custom_validator(value)

            # Handle based on status
            if result.status == 'warning':
                return self._warning(result.message, value)
            elif not result.ok:
                return self._fail(result.message, revert_delay_ms)

        return self._success(value)

    def _success(self, value: str) -> bool:
        """Handle successful validation."""
        self.component.configure(border_color=self.original_border_color)
        self.validation_tooltip.hide()
        self.last_valid_value = value
        return True

    def _warning(self, reason: str, value: str) -> bool:
        """Handle warning state with orange border."""
        self.component.configure(border_color=self.warning_border_color)
        if self.should_show_tooltip():
            self.validation_tooltip.show_warning(reason)
        self.last_valid_value = value

        # Update external validation state if provided
        if self.validation_state:
            self.validation_state.set_warning(reason)

        return True

    def _fail(self, reason: str, revert_delay_ms: int | None) -> bool:
        """Handle validation failure."""
        self.component.configure(border_color=self.error_border_color)
        if self.should_show_tooltip():
            self.validation_tooltip.show_error(reason)

        # Update external validation state if provided
        if self.validation_state:
            self.validation_state.set_error(reason)

        if revert_delay_ms is not None:
            self.revert_after_id = self.component.after(revert_delay_ms, self._do_revert)
        else:
            self._do_revert()
        return False

    def _do_revert(self):
        """Revert to last valid value."""
        self.var.set(self.last_valid_value)
        self.component.configure(border_color=self.original_border_color)
        self.validation_tooltip.hide()

    def debounced_validate(self, *_):
        """Validate with debouncing - called on variable changes."""
        if not self.touched:
            if self.validation_after_id:
                with contextlib.suppress(Exception):
                    self.component.after_cancel(self.validation_after_id)
                self.validation_after_id = None
            return

        # Clear validation UI immediately when field is modified
        self.validation_tooltip.hide()
        self.component.configure(border_color=self.original_border_color)

        # Cancel pending operations
        if self.revert_after_id:
            with contextlib.suppress(Exception):
                self.component.after_cancel(self.revert_after_id)
            self.revert_after_id = None
        if self.validation_after_id:
            with contextlib.suppress(Exception):
                self.component.after_cancel(self.validation_after_id)

        # Schedule validation
        self.validation_after_id = self.component.after(
            COMPONENT_VALIDATION_SETTINGS.debounce_stop_typing_ms,
            lambda: self.validate_value(
                self.var.get(),
                COMPONENT_VALIDATION_SETTINGS.debounced_invalid_revert_ms
            )
        )

    def on_focus_in(self, _e=None):
        """Handle focus in event."""
        self.touched = False
        # Re-show tooltip if there's an existing validation state
        if self.validation_state and self.should_show_tooltip():
            if self.validation_state.status == 'error':
                self.validation_tooltip.show_error(self.validation_state.message, duration_ms=None)
            elif self.validation_state.status == 'warning':
                self.validation_tooltip.show_warning(self.validation_state.message, duration_ms=None)

    def on_user_input(self, _e=None):
        """Handle user input event."""
        self.touched = True

    def on_focus_out(self, _e=None):
        """Handle focus out event - validate if user interacted."""
        # Always hide tooltip on focus out
        self.validation_tooltip.hide()

        if self.touched:
            self.validate_value(
                self.var.get(),
                COMPONENT_VALIDATION_SETTINGS.focusout_invalid_revert_ms
            )

    def cleanup(self):
        """Cleanup resources."""
        if self.validation_after_id:
            with contextlib.suppress(Exception):
                self.component.after_cancel(self.validation_after_id)
        if self.revert_after_id:
            with contextlib.suppress(Exception):
                self.component.after_cancel(self.revert_after_id)


class ModelOutputValidator:
    """
    Smart validator for model output paths with auto-correction.
    Delegates validation logic to validation.py.
    """

    def __init__(
        self,
        var: tk.Variable,
        ui_state: UIState,
        format_var_name: str = "output_model_format",
        method_var_name: str = "training_method",
        prefix_var_name: str = "save_filename_prefix",
    ):
        self.var = var
        self.ui_state = ui_state
        self.format_var = ui_state.get_var(format_var_name)
        self.method_var = ui_state.get_var(method_var_name)
        self.prefix_var = ui_state.get_var(prefix_var_name)
        self.autocorrect_var = ui_state.get_var("validation_auto_correct")
        self.friendly_names_var = ui_state.get_var("use_friendly_names")

        self.state = ValidationState()
        self._trace_ids: dict[str, str] = {}

    def validate(self, value: str) -> ValidationResult:
        """
        Validate model output path using validation.py logic.

        Returns:
            ValidationResult with status (success/warning/error)
        """
        from modules.util.enum.ModelFormat import ModelFormat
        from modules.util.enum.TrainingMethod import TrainingMethod
        from modules.util.ui.validation import validate_destination

        value = value.strip()
        if not value:
            self.state.clear()
            return ValidationResult(ok=True, corrected=None, message="", status='success')

        try:
            output_format = ModelFormat[self.format_var.get()]
        except (KeyError, ValueError):
            output_format = ModelFormat.SAFETENSORS

        try:
            training_method = TrainingMethod[self.method_var.get()]
        except (KeyError, ValueError):
            training_method = TrainingMethod.FINE_TUNE

        try:
            autocorrect = bool(self.autocorrect_var.get())
        except Exception:
            autocorrect = True

        prefix = self.prefix_var.get() if self.prefix_var else ""

        try:
            use_friendly_names = bool(self.friendly_names_var.get())
        except Exception:
            use_friendly_names = False

        # Use validation logic from validation.py
        result = validate_destination(
            value,
            output_format,
            training_method,
            autocorrect=autocorrect,
            prefix=prefix,
            use_friendly_names=use_friendly_names,
            is_output=True
        )

        # Update internal state for hover behavior
        if result.status == 'error':
            self.state.set_error(result.message)
        elif result.status == 'warning':
            self.state.set_warning(result.message)
        else:
            self.state.clear()

        # Apply correction if available
        if result.corrected and result.corrected != value:
            self.var.set(result.corrected)

        return result

    def trigger_validation(self, *_):
        self.validate(self.var.get())

    def on_value_change(self, *_):
        self.state.clear()

    def setup_traces(self):
        self._trace_ids['format'] = self.format_var.trace_add("write", self.trigger_validation)
        self._trace_ids['method'] = self.method_var.trace_add("write", self.trigger_validation)
        self._trace_ids['prefix'] = self.prefix_var.trace_add("write", self.trigger_validation)
        self._trace_ids['friendly_names'] = self.friendly_names_var.trace_add("write", self.trigger_validation)

    def cleanup_traces(self):
        for key, trace_id in self._trace_ids.items():
            with contextlib.suppress(Exception):
                var = getattr(self, f"{key}_var", None)
                if var:
                    var.trace_remove("write", trace_id)
        self._trace_ids.clear()


# UI components

def entry(
        master,
        row,
        column,
        ui_state: UIState,
        var_name: str,
        command: Callable[[], None] = None,
        tooltip: str = "",
        wide_tooltip: bool = False,
        width: int = 140,
        sticky: str = "new",
        custom_validator: Callable[[str], ValidationResult] | None = None,
        validation_state: ValidationState = None,
        enable_hover_validation: bool = False,
):
    """
    Create an entry widget with optional validation.

    Args:
        validation_state: Optional ValidationState object for stateful validators.
                         If provided, enables hover behavior to re-show validation tooltips.
        enable_hover_validation: If True, enables hover behavior to show validation state tooltips.
    """
    var = ui_state.get_var(var_name)
    trace_id = None
    if command:
        trace_id = ui_state.add_var_trace(var_name, command)

    component = ctk.CTkEntry(master, textvariable=var, width=width)
    component.grid(row=row, column=column, padx=PAD, pady=PAD, sticky=sticky)

    # Create validation handler
    validation_handler = EntryValidationHandler(
        component=component,
        var=var,
        var_name=var_name,
        ui_state=ui_state,
        custom_validator=custom_validator,
        validation_state=validation_state,
    )

    # Setup variable trace for validation
    validation_trace_name = var.trace_add("write", validation_handler.debounced_validate)

    # Bind focus and input events
    component.bind("<FocusIn>", validation_handler.on_focus_in)
    component.bind("<Key>", validation_handler.on_user_input)
    component.bind("<<Paste>>", validation_handler.on_user_input)
    component.bind("<<Cut>>", validation_handler.on_user_input)
    component.bind("<FocusOut>", validation_handler.on_focus_out)

    # Optional hover validation behavior
    if enable_hover_validation and validation_state:
        show_tooltips_var = ui_state.get_var("validation_show_tooltips")
        validation_tooltip = validation_handler.validation_tooltip

        def on_hover_enter(_e=None):
            """Re-show tooltip on hover if validation error/warning exists."""
            try:
                show_tooltips = bool(show_tooltips_var.get())
            except Exception:
                show_tooltips = True

            if not show_tooltips:
                return

            # Show tooltip without auto-hide when hovering
            if validation_state.status == 'error':
                validation_tooltip.show_error(validation_state.message, duration_ms=None)
            elif validation_state.status == 'warning':
                validation_tooltip.show_warning(validation_state.message, duration_ms=None)

        def on_hover_leave(_e=None):
            """Hide tooltip when mouse leaves the field."""
            validation_tooltip.hide()

        component.bind("<Enter>", on_hover_enter, add="+")
        component.bind("<Leave>", on_hover_leave, add="+")

    # Override destroy to cleanup resources
    original_destroy = component.destroy

    def new_destroy():
        # 'temporary' fix until https://github.com/TomSchimansky/CustomTkinter/pull/2077 is merged
        # unfortunately Tom has admitted to forgetting about how to maintain CTK so this likely will never be merged
        if component._textvariable_callback_name:
            component._textvariable.trace_remove("write", component._textvariable_callback_name)
            component._textvariable_callback_name = ""

        # Cleanup validation
        validation_handler.cleanup()
        var.trace_remove("write", validation_trace_name)

        if command is not None and trace_id is not None:
            ui_state.remove_var_trace(var_name, trace_id)

        original_destroy()

    component.destroy = new_destroy

    if tooltip:
        ToolTip(component, tooltip, wide=wide_tooltip)

    return component


def model_output_entry(
        master,
        row,
        column,
        ui_state: UIState,
        var_name: str = "output_model_destination",
        format_var_name: str = "output_model_format",
        method_var_name: str = "training_method",
        prefix_var_name: str = "save_filename_prefix",
        command: Callable[[str], None] = None,
):
    """
    Specialized entry for model output paths with smart validation.
    Automatically updates validation when format, method, or prefix changes.
    """
    frame = ctk.CTkFrame(master, fg_color="transparent")
    frame.grid(row=row, column=column, padx=0, pady=0, sticky="new")
    frame.grid_columnconfigure(0, weight=1)

    var = ui_state.get_var(var_name)

    validator = ModelOutputValidator(
        var=var,
        ui_state=ui_state,
        format_var_name=format_var_name,
        method_var_name=method_var_name,
        prefix_var_name=prefix_var_name,
    )

    # Create entry with smart validation and hover behavior
    entry_widget = entry(
        frame, row=0, column=0, ui_state=ui_state, var_name=var_name,
        custom_validator=validator.validate,
        validation_state=validator.state,
        enable_hover_validation=True,
        command=command
    )

    # Setup reactive traces to re-validate when related fields change
    validator.setup_traces()

    # Clear validation state when value changes (before validation runs)
    validation_trace = var.trace_add("write", validator.on_value_change)

    # Cleanup on destroy
    original_destroy = entry_widget.destroy

    def new_destroy():
        with contextlib.suppress(Exception):
            var.trace_remove("write", validation_trace)
        validator.cleanup_traces()
        original_destroy()

    entry_widget.destroy = new_destroy

    # Enable drag-and-drop
    _register_drop_target(entry_widget, ui_state, var_name, command)

    # Browse button
    def open_dialog():
        filetypes = [
            ("All Files", "*.*"),
            ("Diffusers", "model_index.json"),
            ("Checkpoint", "*.ckpt *.pt *.bin"),
            ("Safetensors", "*.safetensors"),
        ]
        selected_path = filedialog.asksaveasfilename(filetypes=filetypes)
        if selected_path:
            var.set(selected_path)
            if command:
                command(selected_path)

    button_component = ctk.CTkButton(frame, text="...", width=40, command=open_dialog)
    button_component.grid(row=0, column=1, padx=(0, PAD), pady=PAD, sticky="nsew")

    return frame


def path_entry(
        master,
        row,
        column,
        ui_state: UIState,
        var_name: str,
        path_modifier: Callable[[str], str] = None,
        is_output: bool = False,
        path_type: str = "file",  # "file" or "directory"
        command: Callable[[str], None] = None,
        allow_model_files: bool = True,
        allow_image_files: bool = False,
        valid_extensions: list[str] = None,
):
    """
    General-purpose path entry component for files and directories.

    For model output destination with smart validation, use model_output_entry() instead.

    Args:
        master: Parent widget
        row, column: Grid position
        ui_state: UI state manager
        var_name: Variable name in ui_state
        path_modifier: Optional function to modify selected paths
        is_output: Whether this is an output path
        path_type: "file" or "directory"
        command: Optional callback when path changes
        allow_model_files: Show model file types in dialog (file mode only)
        allow_image_files: Show image file types in dialog (file mode only)
        valid_extensions: List of valid extensions (file mode only)

    Returns:
        Frame containing entry and browse button
    """
    frame = ctk.CTkFrame(master, fg_color="transparent")
    frame.grid(row=row, column=column, padx=0, pady=0, sticky="new")
    frame.grid_columnconfigure(0, weight=1)

    # Determine if this is an output path
    if not is_output:
        meta = ui_state.get_field_metadata(var_name)
        is_output = getattr(meta, 'is_output', False)

    var = ui_state.get_var(var_name)

    # Simple path validation with auto-trimming for directories
    def simple_path_validator(value: str) -> ValidationResult:
        if not value:
            return validate_file_path(value, is_output, valid_extensions, path_type)

        # Auto-trim whitespace from directory names
        if path_type == "directory":
            path = Path(value)
            trimmed_name = path.name.strip()
            if trimmed_name != path.name:
                sep = '\\' if '\\' in value else '/'
                trimmed_path = f"{path.parent}{sep}{trimmed_name}" if str(path.parent) != '.' else trimmed_name
                var.set(trimmed_path)
                value = trimmed_path

        return validate_file_path(value, is_output, valid_extensions, path_type)

    # Create entry with validation
    entry_widget = entry(
        frame, row=0, column=0, ui_state=ui_state, var_name=var_name,
        custom_validator=simple_path_validator, command=command
    )

    # Enable drag-and-drop
    _register_drop_target(entry_widget, ui_state, var_name, command)

    # Browse button
    def open_dialog():
        if path_type == "directory":
            selected_path = filedialog.askdirectory()
        else:
            # Build file type filters
            filetypes = [("All Files", "*.*")]

            if allow_model_files:
                filetypes.extend([
                    ("Diffusers", "model_index.json"),
                    ("Checkpoint", "*.ckpt *.pt *.bin"),
                    ("Safetensors", "*.safetensors"),
                ])
            if allow_image_files:
                filetypes.extend([
                    ("Image", ' '.join([f"*.{x}" for x in supported_image_extensions()])),
                ])

            if is_output:
                selected_path = filedialog.asksaveasfilename(filetypes=filetypes)
            else:
                selected_path = filedialog.askopenfilename(filetypes=filetypes)

        if selected_path:
            if path_modifier:
                selected_path = path_modifier(selected_path)

            var.set(selected_path)

            if command:
                command(selected_path)

    button_component = ctk.CTkButton(frame, text="...", width=40, command=open_dialog)
    button_component.grid(row=0, column=1, padx=(0, PAD), pady=PAD, sticky="nsew")

    return frame


def file_entry(
        master, row, column, ui_state: UIState, var_name: str,
        path_modifier: Callable[[str], str] = None,
        is_output: bool = False,
        allow_model_files: bool = True,
        allow_image_files: bool = False,
        command: Callable[[str], None] = None,
        valid_extensions: list[str] = None,
        path_type: str = "file",
):
    return path_entry(
        master=master,
        row=row,
        column=column,
        ui_state=ui_state,
        var_name=var_name,
        path_modifier=path_modifier,
        is_output=is_output,
        path_type=path_type,
        command=command,
        allow_model_files=allow_model_files,
        allow_image_files=allow_image_files,
        valid_extensions=valid_extensions,
    )


def dir_entry(
        master, row, column, ui_state: UIState, var_name: str,
        command: Callable[[str], None] = None,
        is_output: bool = False
):
    return path_entry(
        master=master,
        row=row,
        column=column,
        ui_state=ui_state,
        var_name=var_name,
        is_output=is_output,
        path_type="directory",
        command=command,
    )


def app_title(master, row, column):
    frame = ctk.CTkFrame(master)
    frame.grid(row=row, column=column, padx=5, pady=5, sticky="nsew")

    image_component = ctk.CTkImage(
        Image.open("resources/icons/icon.png").resize((40, 40), Image.Resampling.LANCZOS),
        size=(40, 40)
    )
    image_label_component = ctk.CTkLabel(frame, image=image_component, text="")
    image_label_component.grid(row=0, column=0, padx=PAD, pady=PAD)

    label_component = ctk.CTkLabel(frame, text="OneTrainer", font=ctk.CTkFont(size=20, weight="bold"))
    label_component.grid(row=0, column=1, padx=(0, PAD), pady=PAD)


def label(master, row, column, text, pad=PAD, tooltip=None, wide_tooltip=False, wraplength=0):
    component = ctk.CTkLabel(master, text=text, wraplength=wraplength)
    component.grid(row=row, column=column, padx=pad, pady=pad, sticky="nw")
    if tooltip:
        ToolTip(component, tooltip, wide=wide_tooltip)
    return component



def time_entry(master, row, column, ui_state: UIState, var_name: str, unit_var_name, supports_time_units: bool = True):
    frame = ctk.CTkFrame(master, fg_color="transparent")
    frame.grid(row=row, column=column, padx=0, pady=0, sticky="new")

    frame.grid_columnconfigure(0, weight=0)
    frame.grid_columnconfigure(1, weight=1)

    entry(frame, row=0, column=0, ui_state=ui_state, var_name=var_name, width=50)

    values = [str(x) for x in list(TimeUnit)]
    if not supports_time_units:
        values = [str(x) for x in list(TimeUnit) if not x.is_time_unit()]

    unit_component = ctk.CTkOptionMenu(
        frame,
        values=values,
        variable=ui_state.get_var(unit_var_name),
        width=100,
    )
    unit_component.grid(row=0, column=1, padx=(0, PAD), pady=PAD, sticky="new")

    return frame


def icon_button(master, row, column, text, command):
    component = ctk.CTkButton(master, text=text, width=40, command=command)
    component.grid(row=row, column=column, padx=PAD, pady=PAD, sticky="new")
    return component


def button(master, row, column, text, command, tooltip=None, **kwargs):
    # Pop grid-specific parameters from kwargs, using PAD as the default if not provided.
    padx = kwargs.pop('padx', PAD)
    pady = kwargs.pop('pady', PAD)

    component = ctk.CTkButton(master, text=text, command=command, **kwargs)
    component.grid(row=row, column=column, padx=padx, pady=pady, sticky="new")
    if tooltip:
        ToolTip(component, tooltip, x_position=25)
    return component


def options(master, row, column, values, ui_state: UIState, var_name: str, command: Callable[[str], None] = None):
    component = ctk.CTkOptionMenu(master, values=values, variable=ui_state.get_var(var_name), command=command)
    component.grid(row=row, column=column, padx=PAD, pady=(PAD, PAD), sticky="new")

    # temporary fix until https://github.com/TomSchimansky/CustomTkinter/pull/2246 is merged
    def create_destroy(component):
        orig_destroy = component.destroy

        def destroy(self):
            orig_destroy()
            CTkScalingBaseClass.destroy(self)

        return destroy

    destroy = create_destroy(component._dropdown_menu)
    component._dropdown_menu.destroy = lambda: destroy(component._dropdown_menu)

    return component


def options_adv(master, row, column, values, ui_state: UIState, var_name: str,
                command: Callable[[str], None] = None, adv_command: Callable[[], None] = None):
    frame = ctk.CTkFrame(master, fg_color="transparent")
    frame.grid(row=row, column=column, padx=0, pady=0, sticky="new")

    frame.grid_columnconfigure(0, weight=1)

    component = ctk.CTkOptionMenu(frame, values=values, variable=ui_state.get_var(var_name), command=command)
    component.grid(row=0, column=0, padx=PAD, pady=(PAD, PAD), sticky="new")

    button_component = ctk.CTkButton(frame, text="…", width=20, command=adv_command)
    button_component.grid(row=0, column=1, padx=(0, PAD), pady=PAD, sticky="nsew")

    if command:
        command(ui_state.get_var(var_name).get())  # call command once to set the initial value

    # temporary fix until https://github.com/TomSchimansky/CustomTkinter/pull/2246 is merged
    def create_destroy(component):
        orig_destroy = component.destroy

        def destroy(self):
            orig_destroy()
            CTkScalingBaseClass.destroy(self)

        return destroy

    destroy = create_destroy(component._dropdown_menu)
    component._dropdown_menu.destroy = lambda: destroy(component._dropdown_menu)

    return frame, {'component': component, 'button_component': button_component}


def options_kv(master, row, column, values: list[tuple[str, Any]], ui_state: UIState, var_name: str,
               command: Callable[[Any], None] = None):
    var = ui_state.get_var(var_name)
    keys = [key for key, value in values]

    # if the current value is not valid, select the first option
    if var.get() not in [str(value) for key, value in values] and len(keys) > 0:
        var.set(values[0][1])

    deactivate_update_var = False

    def update_component(text):
        for key, value in values:
            if text == key:
                nonlocal deactivate_update_var
                deactivate_update_var = True
                var.set(value)
                if command:
                    command(value)
                deactivate_update_var = False
                break

    component = ctk.CTkOptionMenu(master, values=keys, command=update_component)
    component.grid(row=row, column=column, padx=PAD, pady=(PAD, PAD), sticky="new")

    def update_var():
        if not deactivate_update_var:
            for key, value in values:
                if var.get() == str(value):
                    if component.winfo_exists():  # the component could already be destroyed
                        component.set(key)
                        if command:
                            command(value)
                        break

    var.trace_add("write", lambda _0, _1, _2: update_var())
    update_var()  # call update_var once to set the initial value

    # temporary fix until https://github.com/TomSchimansky/CustomTkinter/pull/2246 is merged
    def create_destroy(component):
        orig_destroy = component.destroy

        def destroy(self):
            orig_destroy()
            CTkScalingBaseClass.destroy(self)

        return destroy

    destroy = create_destroy(component._dropdown_menu)
    component._dropdown_menu.destroy = lambda: destroy(component._dropdown_menu)

    return component


def switch(
        master,
        row,
        column,
        ui_state: UIState,
        var_name: str,
        command: Callable[[], None] = None,
        text: str = "",
):
    var = ui_state.get_var(var_name)
    if command:
        trace_id = ui_state.add_var_trace(var_name, command)

    component = ctk.CTkSwitch(master, variable=var, text=text, command=command)
    component.grid(row=row, column=column, padx=PAD, pady=(PAD, PAD), sticky="new")

    def create_destroy(component):
        orig_destroy = component.destroy

        def destroy(self):
            if command is not None:
                ui_state.remove_var_trace(var_name, trace_id)

            orig_destroy()

        return destroy

    destroy = create_destroy(component)
    component.destroy = lambda: destroy(component)

    return component


def progress(master, row, column):
    component = ctk.CTkProgressBar(master)
    component.grid(row=row, column=column, padx=PAD, pady=(PAD, PAD), sticky="ew")
    return component


def double_progress(master, row, column, label_1, label_2):
    frame = ctk.CTkFrame(master, fg_color="transparent")
    frame.grid(row=row, column=column, padx=0, pady=0, sticky="nsew")

    frame.grid_rowconfigure(0, weight=1)
    frame.grid_rowconfigure(1, weight=1)
    frame.grid_columnconfigure(0, weight=1)

    label_1_component = ctk.CTkLabel(frame, text=label_1)
    label_1_component.grid(row=0, column=0, padx=(PAD, PAD), pady=(0, 0), sticky="new")

    label_2_component = ctk.CTkLabel(frame, text=label_2)
    label_2_component.grid(row=1, column=0, padx=(PAD, PAD), pady=(0, 0), sticky="sew")

    progress_1_component = ctk.CTkProgressBar(frame)
    progress_1_component.grid(row=0, column=1, padx=(PAD, PAD), pady=(PAD, 0), sticky="new")

    progress_2_component = ctk.CTkProgressBar(frame)
    progress_2_component.grid(row=1, column=1, padx=(PAD, PAD), pady=(0, PAD), sticky="sew")

    description_1_component = ctk.CTkLabel(frame, text="")
    description_1_component.grid(row=0, column=2, padx=(PAD, PAD), pady=(0, 0), sticky="new")

    description_2_component = ctk.CTkLabel(frame, text="")
    description_2_component.grid(row=1, column=2, padx=(PAD, PAD), pady=(0, 0), sticky="sew")

    def set_1(value, max_value):
        progress_1_component.set(value / max_value)
        description_1_component.configure(text=f"{value}/{max_value}")

    def set_2(value, max_value):
        progress_2_component.set(value / max_value)
        description_2_component.configure(text=f"{value}/{max_value}")

    return set_1, set_2
