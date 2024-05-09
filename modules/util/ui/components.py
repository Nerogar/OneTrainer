from tkinter import filedialog
from typing import Tuple, Any, Callable

import customtkinter as ctk
from PIL import Image
from customtkinter.windows.widgets.scaling import CTkScalingBaseClass

from modules.util.enum.TimeUnit import TimeUnit
from modules.util.path_util import supported_image_extensions
from modules.util.ui.ToolTip import ToolTip
from modules.util.ui.UIState import UIState

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
        wide_tooltip: bool = False
):
    var = ui_state.get_var(var_name)
    if command:
        trace_id = ui_state.add_var_trace(var_name, command)

    component = ctk.CTkEntry(master, textvariable=var)
    component.grid(row=row, column=column, padx=PAD, pady=PAD, sticky="new")

    def create_destroy(component):
        orig_destroy = component.destroy

        def destroy(self):
            # temporary fix until https://github.com/TomSchimansky/CustomTkinter/pull/2077 is merged
            if self._textvariable_callback_name:
                self._textvariable.trace_remove("write", self._textvariable_callback_name)
                self._textvariable_callback_name = ""

            if command is not None:
                ui_state.remove_var_trace(var_name, trace_id)

            orig_destroy()

        return destroy

    destroy = create_destroy(component)
    component.destroy = lambda: destroy(component)

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

    entry_component = ctk.CTkEntry(frame, textvariable=ui_state.get_var(var_name))
    entry_component.grid(row=0, column=0, padx=(PAD, PAD), pady=PAD, sticky="new")

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

    # temporary fix until https://github.com/TomSchimansky/CustomTkinter/pull/2077 is merged
    def create_destroy(component):
        orig_destroy = component.destroy

        def destroy(self):
            if self._textvariable_callback_name:
                self._textvariable.trace_remove("write", self._textvariable_callback_name)
                self._textvariable_callback_name = ""

            orig_destroy()

        return destroy

    destroy = create_destroy(entry_component)
    entry_component.destroy = lambda: destroy(entry_component)

    button_component = ctk.CTkButton(frame, text="...", width=40, command=__open_dialog)
    button_component.grid(row=0, column=1, padx=(0, PAD), pady=PAD, sticky="nsew")

    return frame


def dir_entry(master, row, column, ui_state: UIState, var_name: str, command: Callable[[str], None] = None):
    frame = ctk.CTkFrame(master, fg_color="transparent")
    frame.grid(row=row, column=column, padx=0, pady=0, sticky="new")

    frame.grid_columnconfigure(0, weight=1)

    entry_component = ctk.CTkEntry(frame, textvariable=ui_state.get_var(var_name))
    entry_component.grid(row=0, column=0, padx=(PAD, PAD), pady=PAD, sticky="new")

    def __open_dialog():
        dir_path = filedialog.askdirectory()

        if dir_path:
            ui_state.get_var(var_name).set(dir_path)

            if command:
                command(dir_path)

    # temporary fix until https://github.com/TomSchimansky/CustomTkinter/pull/2077 is merged
    def create_destroy(component):
        orig_destroy = component.destroy

        def destroy(self):
            if self._textvariable_callback_name:
                self._textvariable.trace_remove("write", self._textvariable_callback_name)
                self._textvariable_callback_name = ""

            orig_destroy()

        return destroy

    destroy = create_destroy(entry_component)
    entry_component.destroy = lambda: destroy(entry_component)

    button_component = ctk.CTkButton(frame, text="...", width=40, command=__open_dialog)
    button_component.grid(row=0, column=1, padx=(0, PAD), pady=PAD, sticky="nsew")

    return frame


def time_entry(master, row, column, ui_state: UIState, var_name: str, unit_var_name, supports_time_units: bool = True):
    frame = ctk.CTkFrame(master, fg_color="transparent")
    frame.grid(row=row, column=column, padx=0, pady=0, sticky="new")

    frame.grid_columnconfigure(0, weight=0)
    frame.grid_columnconfigure(1, weight=1)

    entry_component = ctk.CTkEntry(frame, textvariable=ui_state.get_var(var_name), width=50)
    entry_component.grid(row=0, column=0, padx=PAD, pady=PAD, sticky="new")

    # temporary fix until https://github.com/TomSchimansky/CustomTkinter/pull/2077 is merged
    def create_destroy(component):
        orig_destroy = component.destroy

        def destroy(self):
            if self._textvariable_callback_name:
                self._textvariable.trace_remove("write", self._textvariable_callback_name)
                self._textvariable_callback_name = ""

            orig_destroy()

        return destroy

    destroy = create_destroy(entry_component)
    entry_component.destroy = lambda: destroy(entry_component)

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


def button(master, row, column, text, command, tooltip=None):
    component = ctk.CTkButton(master, text=text, command=command)
    component.grid(row=row, column=column, padx=PAD, pady=PAD, sticky="new")
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
                command: Callable[[str], None] = None, adv_command: Callable[[str], None] = None):
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


def options_kv(master, row, column, values: list[Tuple[str, Any]], ui_state: UIState, var_name: str,
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

    component = ctk.CTkSwitch(master, variable=var, text=text)
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
        description_1_component.configure(text="{0}/{1}".format(value, max_value))

    def set_2(value, max_value):
        progress_2_component.set(value / max_value)
        description_2_component.configure(text="{0}/{1}".format(value, max_value))

    return set_1, set_2
