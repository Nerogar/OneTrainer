from tkinter import filedialog
from typing import Tuple, Any, Callable

import customtkinter as ctk
from PIL import Image

from modules.util.enum.TimeUnit import TimeUnit
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


def label(master, row, column, text, pad=PAD):
    component = ctk.CTkLabel(master, text=text)
    component.grid(row=row, column=column, padx=pad, pady=pad, sticky="nw")
    return component


def entry(master, row, column, ui_state: UIState, var_name: str):
    component = ctk.CTkEntry(master, textvariable=ui_state.vars[var_name])
    component.grid(row=row, column=column, padx=PAD, pady=PAD, sticky="new")
    return component


def file_entry(master, row, column, ui_state: UIState, var_name: str, command: Callable[[str], None] = None):
    frame = ctk.CTkFrame(master, fg_color="transparent")
    frame.grid(row=row, column=column, padx=0, pady=0, sticky="new")

    frame.grid_columnconfigure(0, weight=1)

    entry_component = ctk.CTkEntry(frame, textvariable=ui_state.vars[var_name])
    entry_component.grid(row=0, column=0, padx=(PAD, PAD), pady=PAD, sticky="new")

    def __open_dialog():
        file_path = filedialog.askopenfilename(filetypes=[
            ("All Files", "*.*"),
            ("Diffusers", "model_index.json"),
            ("Checkpoint", "*.ckpt *.pt *.bin"),
            ("Safetensors", "*.safetensors"),
        ])

        if file_path:
            ui_state.vars[var_name].set(file_path)

            if command:
                command(file_path)

    button_component = ctk.CTkButton(frame, text="...", width=40, command=__open_dialog)
    button_component.grid(row=0, column=1, padx=(0, PAD), pady=PAD, sticky="nsew")

    return frame


def dir_entry(master, row, column, ui_state: UIState, var_name: str, command: Callable[[str], None] = None):
    frame = ctk.CTkFrame(master, fg_color="transparent")
    frame.grid(row=row, column=column, padx=0, pady=0, sticky="new")

    frame.grid_columnconfigure(0, weight=1)

    entry_component = ctk.CTkEntry(frame, textvariable=ui_state.vars[var_name])
    entry_component.grid(row=0, column=0, padx=(PAD, PAD), pady=PAD, sticky="new")

    def __open_dialog():
        dir_path = filedialog.askdirectory()

        if dir_path:
            ui_state.vars[var_name].set(dir_path)

            if command:
                command(dir_path)

    button_component = ctk.CTkButton(frame, text="...", width=40, command=__open_dialog)
    button_component.grid(row=0, column=1, padx=(0, PAD), pady=PAD, sticky="nsew")

    return frame


def time_entry(master, row, column, ui_state: UIState, var_name: str, unit_var_name):
    frame = ctk.CTkFrame(master, fg_color="transparent")
    frame.grid(row=row, column=column, padx=0, pady=0, sticky="new")

    frame.grid_columnconfigure(0, weight=1)

    entry_component = ctk.CTkEntry(frame, textvariable=ui_state.vars[var_name])
    entry_component.grid(row=0, column=0, padx=PAD, pady=PAD, sticky="new")

    unit_component = ctk.CTkOptionMenu(
        frame,
        values=[str(x) for x in list(TimeUnit)],
        variable=ui_state.vars[unit_var_name],
        width=100,
    )
    unit_component.grid(row=0, column=1, padx=(0, PAD), pady=PAD, sticky="new")

    return frame


def icon_button(master, row, column, text, command):
    component = ctk.CTkButton(master, text=text, width=40, command=command)
    component.grid(row=row, column=column, padx=PAD, pady=PAD, sticky="new")
    return component


def button(master, row, column, text, command):
    component = ctk.CTkButton(master, text=text, command=command)
    component.grid(row=row, column=column, padx=PAD, pady=PAD, sticky="new")
    return component


def options(master, row, column, values, ui_state: UIState, var_name: str, command: Callable[[str], None] = None):
    component = ctk.CTkOptionMenu(master, values=values, variable=ui_state.vars[var_name], command=command)
    component.grid(row=row, column=column, padx=PAD, pady=(PAD, PAD), sticky="new")
    return component


def options_kv(master, row, column, values: list[Tuple[str, Any]], ui_state: UIState, var_name: str,
               command: Callable[[None], None] = None):
    var = ui_state.vars[var_name]
    keys = [key for key, value in values]

    def update_component(text):
        for key, value in values:
            if text == key:
                var.set(value)
                if command:
                    command(value)
                break

    component = ctk.CTkOptionMenu(master, values=keys, command=update_component)
    component.grid(row=row, column=column, padx=PAD, pady=(PAD, PAD), sticky="new")

    def update_var():
        for key, value in values:
            if var.get() == str(value):
                component.set(key)
                if command:
                    command(value)
                break

    var.trace_add("write", lambda _0, _1, _2: update_var())
    update_var()  # call update_var once to set the initial value

    return component


def switch(master, row, column, ui_state: UIState, var_name: str):
    component = ctk.CTkSwitch(master, variable=ui_state.vars[var_name], text="")
    component.grid(row=row, column=column, padx=PAD, pady=(PAD, PAD), sticky="new")
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
