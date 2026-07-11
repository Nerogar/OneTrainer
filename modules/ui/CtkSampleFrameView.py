from modules.ui.BaseSampleFrameView import BaseSampleFrameView
from modules.ui.SampleFrameController import SampleFrameController
from modules.util.ui import ctk_components

import customtkinter as ctk


class CtkSampleFrameView(BaseSampleFrameView, ctk.CTkFrame):
    def __init__(
            self,
            parent,
            controller: SampleFrameController,
            ui_state,
            include_prompt: bool = True,
            include_settings: bool = True,
    ):
        ctk.CTkFrame.__init__(self, parent, fg_color="transparent")
        BaseSampleFrameView.__init__(self, ctk_components)

        if include_prompt and include_prompt:
            self.grid_rowconfigure(0, weight=0)
            self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        top_frame = None
        if include_prompt:
            top_frame = ctk.CTkFrame(self, fg_color="transparent")
            top_frame.grid(row=0, column=0, padx=0, pady=0, sticky="nsew")

            top_frame.grid_columnconfigure(0, weight=0)
            top_frame.grid_columnconfigure(1, weight=1)

        bottom_frame = None
        if include_settings:
            bottom_frame = ctk.CTkFrame(self, fg_color="transparent")
            bottom_frame.grid(row=1, column=0, padx=0, pady=0, sticky="nsew")

            bottom_frame.grid_columnconfigure(0, weight=0)
            bottom_frame.grid_columnconfigure(1, weight=1)
            bottom_frame.grid_columnconfigure(2, weight=0)
            bottom_frame.grid_columnconfigure(3, weight=1)

        self.build_content(top_frame, bottom_frame, ui_state, controller, include_prompt, include_settings)
