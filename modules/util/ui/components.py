import contextlib
from collections.abc import Callable
from tkinter import filedialog
from typing import Any

from modules.util.enum.TimeUnit import TimeUnit
from modules.util.path_util import supported_image_extensions
from modules.util.ui.ToolTip import ToolTip
from modules.util.ui.UIState import UIState

import customtkinter as ctk
from customtkinter.windows.widgets.scaling import CTkScalingBaseClass
from PIL import Image

PAD = 10


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
):
    var = ui_state.get_var(var_name)
    trace_id = None
    if command:
        trace_id = ui_state.add_var_trace(var_name, command)

    component = ctk.CTkEntry(master, textvariable=var, width=width)
    component.grid(row=row, column=column, padx=PAD, pady=PAD, sticky=sticky)

    try:
        original_border_color = component.cget("border_color")
    except Exception:
        original_border_color = "gray50"

    error_border_color = "#dc3545"

    validation_after_id = None
    revert_after_id = None
    touched = False

    DEBOUNCE_STOP_TYPING_MS = 1500
    DEBOUNCED_INVALID_REVERT_MS = 1000
    FOCUSOUT_INVALID_REVERT_MS = 1200

    last_valid_value = var.get()

    def validate_value(value: str, revert_delay_ms: int | None) -> bool:
        nonlocal revert_after_id, last_valid_value
        meta = ui_state.get_field_metadata(var_name)
        declared_type = meta.type
        nullable = meta.nullable
        default_val = meta.default

        if revert_after_id:
            with contextlib.suppress(Exception):
                component.after_cancel(revert_after_id)
            revert_after_id = None

        def success():
            nonlocal last_valid_value
            component.configure(border_color=original_border_color)
            last_valid_value = value
            return True

        def do_revert():
            var.set(last_valid_value)
            component.configure(border_color=original_border_color)

        def fail(_reason: str):
            nonlocal revert_after_id
            component.configure(border_color=error_border_color)
            if revert_delay_ms is not None:
                revert_after_id = component.after(revert_delay_ms, do_revert)
            else:
                do_revert()
            return False

        if value == "":
            if nullable:
                return success()
            if declared_type is str:
                if default_val == "":
                    return success()
                return fail("Value required")

        try:
            if declared_type is int:
                int(value)
            elif declared_type is float:
                float(value)
            elif declared_type is bool:
                if value.lower() not in ("true", "false", "0", "1"):
                    return fail("Invalid bool")
            return success()
        except ValueError:
            return fail("Invalid value")

    def debounced_validate(*_):
        nonlocal validation_after_id, revert_after_id
        if not touched:
            if validation_after_id:
                with contextlib.suppress(Exception):
                    component.after_cancel(validation_after_id)
                validation_after_id = None
            return
        if revert_after_id:
            with contextlib.suppress(Exception):
                component.after_cancel(revert_after_id)
            revert_after_id = None
        if validation_after_id:
            with contextlib.suppress(Exception):
                component.after_cancel(validation_after_id)
        validation_after_id = component.after(
            DEBOUNCE_STOP_TYPING_MS,
            lambda: validate_value(var.get(), DEBOUNCED_INVALID_REVERT_MS)
        )

    validation_trace_name = var.trace_add("write", debounced_validate)

    def on_focus_in(_e=None):
        nonlocal touched
        touched = False

    def on_user_input(_e=None):
        nonlocal touched
        touched = True

    def on_focus_out(_e=None):
        # only validate on focus-out if the user interacted with the field.
        if touched:
            validate_value(var.get(), FOCUSOUT_INVALID_REVERT_MS)

    component.bind("<FocusIn>", on_focus_in)
    component.bind("<Key>", on_user_input)
    component.bind("<<Paste>>", on_user_input)
    component.bind("<<Cut>>", on_user_input)
    component.bind("<FocusOut>", on_focus_out)

    original_destroy = component.destroy

    def new_destroy():
        # 'temporary' fix until https://github.com/TomSchimansky/CustomTkinter/pull/2077 is merged
        # unfortunately Tom has admitted to forgetting about how to maintain CTK so this likely will never be merged
        nonlocal validation_after_id, revert_after_id
        if component._textvariable_callback_name:
            component._textvariable.trace_remove("write", component._textvariable_callback_name)
            component._textvariable_callback_name = ""

        if validation_after_id:
            with contextlib.suppress(Exception):
                component.after_cancel(validation_after_id)
        if revert_after_id:
            with contextlib.suppress(Exception):
                component.after_cancel(revert_after_id)

        var.trace_remove("write", validation_trace_name)

        if command is not None and trace_id is not None:
            ui_state.remove_var_trace(var_name, trace_id)

        original_destroy()

    component.destroy = new_destroy

    if tooltip:
        ToolTip(component, tooltip, wide=wide_tooltip)

    return component


def file_entry(
        master, row, column, ui_state: UIState, var_name: str,
        is_output: bool = False,
        path_modifier: Callable[[str], str] = None,
        allow_model_files: bool = True,
        allow_image_files: bool = False,
        command: Callable[[str], None] = None,
):
    frame = ctk.CTkFrame(master, fg_color="transparent")
    frame.grid(row=row, column=column, padx=0, pady=0, sticky="new")

    frame.grid_columnconfigure(0, weight=1)

    entry(frame,row=0, column=0, ui_state=ui_state, var_name=var_name)

    def __open_dialog():
        filetypes = [
            ("All Files", "*.*"),
        ]

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
            file_path = filedialog.asksaveasfilename(filetypes=filetypes)
        else:
            file_path = filedialog.askopenfilename(filetypes=filetypes)

        if file_path:
            if path_modifier:
                file_path = path_modifier(file_path)

            ui_state.get_var(var_name).set(file_path)

            if command:
                command(file_path)

    button_component = ctk.CTkButton(frame, text="...", width=40, command=__open_dialog)
    button_component.grid(row=0, column=1, padx=(0, PAD), pady=PAD, sticky="nsew")

    return frame


def dir_entry(master, row, column, ui_state: UIState, var_name: str, command: Callable[[str], None] = None):
    frame = ctk.CTkFrame(master, fg_color="transparent")
    frame.grid(row=row, column=column, padx=0, pady=0, sticky="new")

    frame.grid_columnconfigure(0, weight=1)

    entry(frame, row=0, column=0, ui_state=ui_state, var_name=var_name)

    def __open_dialog():
        dir_path = filedialog.askdirectory()

        if dir_path:
            ui_state.get_var(var_name).set(dir_path)

            if command:
                command(dir_path)

    button_component = ctk.CTkButton(frame, text="...", width=40, command=__open_dialog)
    button_component.grid(row=0, column=1, padx=(0, PAD), pady=PAD, sticky="nsew")

    return frame


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

def layer_filter_entry(master, row, column, ui_state: UIState, preset_var_name: str, preset_label: str, preset_tooltip: str, presets, entry_var_name, entry_tooltip: str, regex_var_name, regex_tooltip: str, frame_color=None):
    frame = ctk.CTkFrame(master=master, corner_radius=5, fg_color=frame_color)
    frame.grid(row=row, column=column, padx=5, pady=5, sticky="nsew")
    frame.grid_columnconfigure(0, weight=1)

    layer_entry = entry(
        frame, 1, 0, ui_state, entry_var_name,
        tooltip=entry_tooltip
    )
    layer_entry_fg_color = layer_entry.cget("fg_color")
    layer_entry_text_color = layer_entry.cget("text_color")

    regex_label = label(
        frame, 2, 0, "Use Regex",
        tooltip=regex_tooltip,
    )
    regex_switch = switch(
        frame, 2, 1, ui_state, regex_var_name
    )

    # Let the user set their own layer filter
    # TODO
    #if self.train_config.layer_filter and self.train_config.layer_filter_preset == "custom":
    #    self.prior_custom = self.train_config.layer_filter
    #else:
    #    self.prior_custom = ""

    layer_entry.grid_configure(columnspan=2, sticky="ew")

    presets_list = list(presets.keys()) + ["custom"]


    def hide_layer_entry():
        if layer_entry and layer_entry.winfo_manager():
            layer_entry.grid_remove()

    def show_layer_entry():
        if layer_entry and not layer_entry.winfo_manager():
            layer_entry.grid()


    def preset_set_layer_choice(selected: str):
        if not selected or selected not in presets_list:
            selected = presets_list[0]

        if selected == "custom":
            # Allow editing + regex toggle
            show_layer_entry()
            layer_entry.configure(state="normal", fg_color=layer_entry_fg_color, text_color=layer_entry_text_color)
            #layer_entry.cget('textvariable').set("")
            regex_label.grid()
            regex_switch.grid()
        else:
            # Preserve custom text before overwriting
            #if self.prior_selected == "custom":
            #    self.prior_custom = self.layer_entry.get()

            # Resolve preset definition (list[str] OR {'patterns': [...], 'regex': bool})
            preset_def = presets.get(selected, [])
            if isinstance(preset_def, dict):
                patterns = preset_def.get("patterns", [])
                preset_uses_regex = bool(preset_def.get("regex", False))
            else:
                patterns = preset_def
                preset_uses_regex = False

            disabled_color = ("gray85", "gray17")
            disabled_text_color = ("gray30", "gray70")
            layer_entry.configure(state="disabled", fg_color=disabled_color, text_color=disabled_text_color)
            layer_entry.cget('textvariable').set(",".join(patterns))

            ui_state.get_var(entry_var_name).set(",".join(patterns))
            ui_state.get_var(regex_var_name).set(preset_uses_regex)

            regex_label.grid_remove()
            regex_switch.grid_remove()

            if selected == "full" and not patterns:
                hide_layer_entry()
            else:
                show_layer_entry()

#        self.prior_selected = selected

    label(frame, 0, 0, preset_label,
                     tooltip=preset_tooltip)


    ui_state.remove_all_var_traces(preset_var_name)

    layer_selector = options(
        frame, 0, 1, presets_list, ui_state, preset_var_name,
        command=preset_set_layer_choice
    )

    def on_layer_filter_preset_change():
        if not layer_selector:
            return
        selected = ui_state.get_var(preset_var_name).get()
        preset_set_layer_choice(selected)

    ui_state.add_var_trace(
        preset_var_name,
        on_layer_filter_preset_change,
    )

    preset_set_layer_choice(layer_selector.get())

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
