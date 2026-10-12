

from modules.ui.BaseModelTabView import BaseModelTabView
from modules.ui.ModelTabController import ModelTabController
from modules.util.ui import ctk_components

import customtkinter as ctk


class CtkModelTabView(BaseModelTabView):
    def __init__(self, master, controller: ModelTabController, ui_state):
        BaseModelTabView.__init__(self, ctk_components)
        self.master = master
        self.controller = controller
        self.ui_state = ui_state

        master.grid_rowconfigure(0, weight=1)
        master.grid_columnconfigure(0, weight=1)

        self.scroll_frame = None

        self.refresh_ui()

    def _make_svd_frames(self, parent, row: int):
        svd_label_frame = ctk.CTkFrame(parent, fg_color="transparent")
        svd_label_frame.grid(row=row, column=3, sticky="nsew")
        svd_entry_frame = ctk.CTkFrame(parent, fg_color="transparent")
        svd_entry_frame.grid(row=row, column=4, sticky="nsew")
        return svd_label_frame, svd_entry_frame

    def refresh_ui(self):
        if self.scroll_frame:
            self.scroll_frame.destroy()

        self.scroll_frame = ctk.CTkScrollableFrame(self.master, fg_color="transparent")
        self.scroll_frame.grid(row=0, column=0, sticky="nsew")
        self.scroll_frame.grid_columnconfigure(0, weight=1)

        base_frame = ctk.CTkFrame(master=self.scroll_frame, corner_radius=5)
        base_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        base_frame.grid_columnconfigure(0, weight=0)
        base_frame.grid_columnconfigure(1, weight=10)  # , minsize=500)
        base_frame.grid_columnconfigure(2, minsize=50)
        base_frame.grid_columnconfigure(3, weight=0)
        base_frame.grid_columnconfigure(4, weight=1)

        self.build_content(base_frame, self.controller, self.ui_state)
