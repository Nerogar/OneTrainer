from collections.abc import Callable
from tkinter import filedialog

from modules.ui.BaseTopBarView import BaseTopBarView
from modules.ui.TopBarController import TopBarController
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.ui import ctk_components

import customtkinter as ctk


class CtkTopBarView(BaseTopBarView):
    def __init__(
            self,
            master,
            controller: TopBarController,
            ui_state,
            change_model_type_callback: Callable[[ModelType], None],
            change_training_method_callback: Callable[[TrainingMethod], None],
            load_preset_callback: Callable[[], None],
    ):
        BaseTopBarView.__init__(self, ctk_components)

        frame = ctk.CTkFrame(master=master, corner_radius=0)
        frame.grid(row=0, column=0, sticky="nsew")

        self.build(frame, master, controller, ui_state, change_model_type_callback, change_training_method_callback, load_preset_callback)

    def _setup_frame_column_weight(self):
        self.frame.grid_columnconfigure(5, weight=1)

    def _forget_dropdown(self, widget):
        widget.destroy()

    def _show_save_dialog(self, initial_dir: str, callback):
        chosen = filedialog.asksaveasfilename(
            initialdir=initial_dir, defaultextension=".json",
            filetypes=[("JSON", "*.json")],
        )
        if chosen:
            callback(chosen)

    def _show_open_dialog(self, initial_dir: str, callback):
        chosen = filedialog.askopenfilename(
            initialdir=initial_dir, filetypes=[("JSON", "*.json")],
        )
        if chosen:
            callback(chosen)
