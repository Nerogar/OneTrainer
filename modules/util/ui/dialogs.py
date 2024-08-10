from typing import Callable

import customtkinter as ctk


class StringInputDialog(ctk.CTkToplevel):
    def __init__(
            self,
            parent,
            title: str,
            question: str,
            callback: Callable[[str], None],
            default_value: str = None,
            validate_callback: Callable[[str], bool] = None,
            *args, **kwargs
    ):
        ctk.CTkToplevel.__init__(self, parent, *args, **kwargs)
        self.parent = parent

        self.callback = callback
        self.validate_callback = validate_callback

        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        self.title(title)
        self.geometry("300x120")
        self.resizable(False, False)
        self.wait_visibility()
        self.grab_set()
        self.focus_set()

        self.question_label = ctk.CTkLabel(self, text=question)
        self.question_label.grid(row=0, column=0, columnspan=2, sticky="we", padx=5, pady=5)

        self.entry = ctk.CTkEntry(self, width=150)
        self.entry.grid(row=1, column=0, columnspan=2, sticky="we", padx=10, pady=5)

        self.ok_button = ctk.CTkButton(self, width=30, text="ok", command=self.ok)
        self.ok_button.grid(row=2, column=0, sticky="we", padx=10, pady=5)

        self.ok_button = ctk.CTkButton(self, width=30, text="cancel", command=self.cancel)
        self.ok_button.grid(row=2, column=1, sticky="we", padx=10, pady=5)

        if default_value is not None:
            self.entry.insert(0, default_value)

    def ok(self):
        if self.validate_callback is None or self.validate_callback(self.entry.get()):
            self.callback(self.entry.get())
            self.destroy()

    def cancel(self):
        self.destroy()
