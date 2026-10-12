from modules.ui.BaseSampleParamsWindowView import BaseSampleParamsWindowView
from modules.ui.CtkSampleFrameView import CtkSampleFrameView
from modules.ui.SampleFrameController import SampleFrameController
from modules.ui.SampleParamsWindowController import SampleParamsWindowController
from modules.util.ui import ctk_components
from modules.util.ui.ui_utils import set_window_icon

import customtkinter as ctk


class CtkSampleParamsWindowView(BaseSampleParamsWindowView, ctk.CTkToplevel):
    def __init__(self, parent, controller: SampleParamsWindowController, ui_state, *args, **kwargs):
        ctk.CTkToplevel.__init__(self, parent, *args, **kwargs)
        BaseSampleParamsWindowView.__init__(self, ctk_components)

        self.title("Sample")
        self.geometry("800x500")
        self.resizable(True, True)

        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)
        self.grid_columnconfigure(0, weight=1)

        frame = CtkSampleFrameView(self, SampleFrameController(controller.sample, controller.model_type), ui_state)
        frame.grid(row=0, column=0, padx=0, pady=0, sticky="nsew")

        self.components.button(self, 1, 0, "ok", self.destroy)

        self.wait_visibility()
        self.grab_set()
        self.focus_set()
        self.after(200, lambda: set_window_icon(self))
