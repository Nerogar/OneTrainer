from modules.ui.SampleFrame import SampleFrame
from modules.util.config.SampleConfig import SampleConfig
from modules.util.ui import components
from modules.util.ui.ui_utils import set_window_icon
from modules.util.ui.UIState import UIState

import customtkinter as ctk


class SampleParamsWindow(ctk.CTkToplevel):
    def __init__(self, parent, sample: SampleConfig, ui_state: UIState, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        self.sample = sample
        self.ui_state = ui_state

        self.title("Sample")
        self.geometry("800x500")
        self.resizable(True, True)

        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)
        self.grid_columnconfigure(0, weight=1)

        frame = SampleFrame(self, self.sample, self.ui_state)
        frame.grid(row=0, column=0, padx=0, pady=0, sticky="nsew")

        components.button(self, 1, 0, "ok", self.__ok)

        self.wait_visibility()
        self.grab_set()
        self.focus_set()
        self.after(200, lambda: set_window_icon(self))


    def __ok(self):
        self.destroy()
