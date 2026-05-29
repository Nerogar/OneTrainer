from modules.ui.BaseMuonAdamWindowView import BaseMuonAdamWindowView
from modules.ui.MuonAdamWindowController import MuonAdamWindowController
from modules.util.ui import ctk_components
from modules.util.ui.ui_utils import set_window_icon

import customtkinter as ctk


class CtkMuonAdamWindowView(BaseMuonAdamWindowView, ctk.CTkToplevel):
    def __init__(self, parent, controller: MuonAdamWindowController, ui_state, *args, **kwargs):
        ctk.CTkToplevel.__init__(self, parent, *args, **kwargs)
        BaseMuonAdamWindowView.__init__(self, ctk_components)

        self.title(controller.get_title())
        self.geometry("800x500")
        self.resizable(True, True)

        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)
        self.grid_columnconfigure(0, weight=1)

        frame = ctk.CTkScrollableFrame(self, fg_color="transparent")
        frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        frame.grid_columnconfigure(0, weight=0)
        frame.grid_columnconfigure(1, weight=1)
        frame.grid_columnconfigure(2, minsize=50)
        frame.grid_columnconfigure(3, weight=0)
        frame.grid_columnconfigure(4, weight=1)

        self.components.button(self, 1, 0, "ok", command=self.destroy)
        self.build_content(frame, controller, ui_state)

        self.wait_visibility()
        self.grab_set()
        self.focus_set()
        self.after(200, lambda: set_window_icon(self))
