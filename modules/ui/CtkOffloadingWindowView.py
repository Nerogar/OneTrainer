from modules.ui.BaseOffloadingWindowView import BaseOffloadingWindowView
from modules.ui.OffloadingWindowController import OffloadingWindowController
from modules.util.ui import ctk_components
from modules.util.ui.ui_utils import set_window_icon

import customtkinter as ctk


class CtkOffloadingWindowView(BaseOffloadingWindowView, ctk.CTkToplevel):
    def __init__(self, parent, controller: OffloadingWindowController, ui_state, *args, **kwargs):
        ctk.CTkToplevel.__init__(self, parent, *args, **kwargs)
        BaseOffloadingWindowView.__init__(self, ctk_components)

        self.title("Offloading")
        self.geometry("800x400")
        self.resizable(True, True)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        frame = ctk.CTkScrollableFrame(self, fg_color="transparent")
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_columnconfigure(1, weight=1)
        self.build_content(frame, controller, ui_state)
        frame.pack(fill="both", expand=1)
        frame.grid(row=0, column=0, sticky='nsew')
        self.components.button(self, 1, 0, "ok", self.destroy)

        self.wait_visibility()
        self.grab_set()
        self.focus_set()
        self.after(200, lambda: set_window_icon(self))
