import contextlib
import tkinter as tk
from tkinter import filedialog

from modules.ui.BaseGenerateCaptionsWindowView import BaseGenerateCaptionsWindowView
from modules.ui.GenerateCaptionsWindowController import GenerateCaptionsWindowController
from modules.util.ui.ui_utils import set_window_icon

import customtkinter as ctk


class CtkGenerateCaptionsWindowView(BaseGenerateCaptionsWindowView, ctk.CTkToplevel):
    def __init__(self, parent, controller: GenerateCaptionsWindowController, path, parent_include_subdirectories, *args, **kwargs):
        ctk.CTkToplevel.__init__(self, parent, *args, **kwargs)

        if path is None:
            path = ""

        self.controller = controller

        self.mode_var = ctk.StringVar(self, "Create if absent")
        self.modes = ["Replace all captions", "Create if absent", "Add as new line"]
        self.model_var = ctk.StringVar(self, "Blip")
        self.models = ["Blip", "Blip2", "WD14 VIT v2"]

        self.title("Batch generate captions")
        self.geometry("360x360")
        self.resizable(True, True)

        self.frame = ctk.CTkFrame(self, width=600, height=300)
        self.frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        self.model_label = ctk.CTkLabel(self.frame, text="Model", width=100)
        self.model_label.grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.model_dropdown = ctk.CTkOptionMenu(self.frame, variable=self.model_var, values=self.models, dynamic_resizing=False, width=200)
        self.model_dropdown.grid(row=0, column=1, sticky="w", padx=5, pady=5)

        self.path_label = ctk.CTkLabel(self.frame, text="Folder", width=100)
        self.path_label.grid(row=1, column=0, sticky="w",padx=5, pady=5)
        self.path_entry = ctk.CTkEntry(self.frame, width=150)
        self.path_entry.insert(0, path)
        self.path_entry.grid(row=1, column=1, sticky="w", padx=5, pady=5)
        self.path_button = ctk.CTkButton(self.frame, width=30, text="...", command=lambda: self.browse_for_path(self.path_entry))
        self.path_button.grid(row=1, column=1, sticky="e", padx=5, pady=5)

        self.caption_label = ctk.CTkLabel(self.frame, text="Initial Caption", width=100)
        self.caption_label.grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.caption_entry = ctk.CTkEntry(self.frame, width=200)
        self.caption_entry.grid(row=2, column=1, sticky="w", padx=5, pady=5)

        self.prefix_label = ctk.CTkLabel(self.frame, text="Caption Prefix", width=100)
        self.prefix_label.grid(row=3, column=0, sticky="w", padx=5, pady=5)
        self.prefix_entry = ctk.CTkEntry(self.frame, width=200)
        self.prefix_entry.grid(row=3, column=1, sticky="w", padx=5, pady=5)

        self.postfix_label = ctk.CTkLabel(self.frame, text="Caption Postfix", width=100)
        self.postfix_label.grid(row=4, column=0, sticky="w", padx=5, pady=5)
        self.postfix_entry = ctk.CTkEntry(self.frame, width=200)
        self.postfix_entry.grid(row=4, column=1, sticky="w", padx=5, pady=5)

        self.mode_label = ctk.CTkLabel(self.frame, text="Mode", width=100)
        self.mode_label.grid(row=5, column=0, sticky="w", padx=5, pady=5)
        self.mode_dropdown = ctk.CTkOptionMenu(self.frame, variable=self.mode_var, values=self.modes, dynamic_resizing=False, width=200)
        self.mode_dropdown.grid(row=5, column=1, sticky="w", padx=5, pady=5)

        self.include_subdirectories_label = ctk.CTkLabel(self.frame, text="Include subfolders", width=100)
        self.include_subdirectories_label.grid(row=6, column=0, sticky="w", padx=5, pady=5)
        self.include_subdirectories_var = ctk.BooleanVar(self, parent_include_subdirectories)
        self.include_subdirectories_switch = ctk.CTkSwitch(self.frame, text="", variable=self.include_subdirectories_var)
        self.include_subdirectories_switch.grid(row=6, column=1, sticky="w", padx=5, pady=5)

        self.progress_label = ctk.CTkLabel(self.frame, text="Progress: 0/0", width=100)
        self.progress_label.grid(row=7, column=0, sticky="w", padx=5, pady=5)
        self.progress = ctk.CTkProgressBar(self.frame, orientation="horizontal", mode="determinate", width=200)
        self.progress.grid(row=7, column=1, sticky="w", padx=5, pady=5)

        self.create_captions_button = ctk.CTkButton(self.frame, text="Create Captions", width=310, command=self._on_create_captions)
        self.create_captions_button.grid(row=8, column=0, columnspan=2, sticky="w", padx=5, pady=5)

        self.frame.pack(fill="both", expand=True)

        self.wait_visibility()
        self.grab_set()
        self.focus_set()
        self.after(200, lambda: set_window_icon(self))

    def browse_for_path(self, entry_box):
        # get the path from the user
        path = filedialog.askdirectory()
        # set the path to the entry box
        # delete entry box text
        entry_box.focus_set()
        entry_box.delete(0, filedialog.END)
        entry_box.insert(0, path)
        self.focus_set()

    def set_progress(self, value, max_value):
        progress = value / max_value
        self.progress.set(progress)
        self.progress_label.configure(text=f"{value}/{max_value}")
        self.progress.update()

    def _on_create_captions(self):
        self.controller.create_captions(
            model_name=self.model_var.get(),
            path=self.path_entry.get(),
            initial_caption=self.caption_entry.get(),
            caption_prefix=self.prefix_entry.get(),
            caption_postfix=self.postfix_entry.get(),
            mode_str=self.mode_var.get(),
            include_subdirectories=self.include_subdirectories_var.get(),
        )

    def destroy(self):
        with contextlib.suppress(tk.TclError):
            self.grab_release()

        super().destroy()
