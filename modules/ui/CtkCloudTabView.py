

from modules.ui.BaseCloudTabView import BaseCloudTabView
from modules.ui.CloudTabController import CloudTabController
from modules.util.ui import ctk_components

import customtkinter as ctk


class CtkCloudTabView(BaseCloudTabView):
    def __init__(self, master, controller: CloudTabController, ui_state):
        BaseCloudTabView.__init__(self, ctk_components, controller)
        self.master = master
        self.ui_state = ui_state

        self.frame = ctk.CTkScrollableFrame(master, fg_color="transparent")
        self.frame.grid_columnconfigure(0, weight=0)
        self.frame.grid_columnconfigure(1, weight=1)
        self.frame.grid_columnconfigure(2, weight=0)
        self.frame.grid_columnconfigure(3, weight=1)
        self.frame.grid_columnconfigure(4, weight=0)
        self.frame.grid_columnconfigure(5, weight=1)

        self.build_content(self.frame, controller, ui_state)

        self.frame.pack(fill="both", expand=1)


    def _on_set_gpu_types(self):
        self.gpu_types_menu.configure(values=self.controller.get_gpu_types())

    def _make_reattach_frame(self, frame):
        reattach_frame = ctk.CTkFrame(frame, fg_color="transparent")
        reattach_frame.grid(row=9, column=3, padx=0, pady=0, sticky="new")
        reattach_frame.grid_columnconfigure(0, weight=1)
        reattach_frame.grid_columnconfigure(1, weight=1)
        return reattach_frame

    def _make_create_frame(self, frame):
        create_frame = ctk.CTkFrame(frame, fg_color="transparent")
        create_frame.grid(row=1, column=5, padx=0, pady=0, sticky="new")
        create_frame.grid_columnconfigure(0, weight=0)
        create_frame.grid_columnconfigure(1, weight=1)
        return create_frame
