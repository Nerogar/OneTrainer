
from modules.ui.BaseLoraTabView import BaseLoraTabView
from modules.ui.LoraTabController import LoraTabController
from modules.util.enum.ModelType import PeftType
from modules.util.ui import ctk_components

import customtkinter as ctk


class CtkLoraTabView(BaseLoraTabView):
    def __init__(self, master, controller: LoraTabController, ui_state):
        BaseLoraTabView.__init__(self, ctk_components)
        self.master = master
        self.controller = controller
        self.ui_state = ui_state
        self.scroll_frame = None
        self.options_frame = None
        self.refresh_ui()

    def refresh_ui(self):
        if self.scroll_frame:
            self.scroll_frame.destroy()
        self.scroll_frame = ctk.CTkFrame(self.master, fg_color="transparent")
        self.scroll_frame.grid(row=0, column=0, sticky="nsew")
        self.scroll_frame.grid_columnconfigure(0, weight=0)
        self.scroll_frame.grid_columnconfigure(1, weight=1)
        self.scroll_frame.grid_columnconfigure(2, weight=2)
        self.build(self.scroll_frame, self.controller, self.ui_state, self.setup_lora)

    def setup_lora(self, peft_type: PeftType):
        if self.options_frame:
            self.options_frame.destroy()
        self.options_frame = ctk.CTkFrame(self.scroll_frame, fg_color="transparent")
        self.options_frame.grid(row=1, column=0, columnspan=3, sticky="nsew")
        master = self.options_frame
        master.grid_columnconfigure(0, weight=0, uniform="a")
        master.grid_columnconfigure(1, weight=1, uniform="a")
        master.grid_columnconfigure(2, minsize=50, uniform="a")
        master.grid_columnconfigure(3, weight=0, uniform="a")
        master.grid_columnconfigure(4, weight=1, uniform="a")
        self.build_lora_options(master, self.controller, self.ui_state, peft_type)
