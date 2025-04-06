from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.GradientCheckpointingMethod import (
    GradientCheckpointingMethod,
)
from modules.util.ui import components
from modules.util.ui.ui_utils import set_window_icon
from modules.util.ui.UIState import UIState

import customtkinter as ctk


class OffloadingWindow(ctk.CTkToplevel):
    def __init__(
            self,
            parent,
            config: TrainConfig,
            ui_state: UIState,
            *args, **kwargs,
    ):
        super().__init__(parent, *args, **kwargs)

        self.config = config
        self.ui_state = ui_state
        self.image_preview_file_index = 0
        self.ax = None
        self.canvas = None

        self.title("Offloading")
        self.geometry("800x400")
        self.resizable(True, True)

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        frame = self.__content_frame(self)
        frame.grid(row=0, column=0, sticky='nsew')
        components.button(self, 1, 0, "ok", self.__ok)

        self.wait_visibility()
        self.grab_set()
        self.focus_set()
        self.after(200, lambda: set_window_icon(self))


    def __content_frame(self, master):
        frame = ctk.CTkScrollableFrame(master, fg_color="transparent")
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_columnconfigure(1, weight=1)

        # timestep distribution
        components.label(frame, 0, 0, "Gradient checkpointing",
                         tooltip="Enables gradient checkpointing. This reduces memory usage, but increases training time")
        components.options(frame, 0, 1, [str(x) for x in list(GradientCheckpointingMethod)], self.ui_state,
                           "gradient_checkpointing")

        # gradient checkpointing layer offloading
        components.label(frame, 1, 0, "Async Offloading",
                         tooltip="Enables Asynchronous offloading.")
        components.switch(frame, 1, 1, self.ui_state, "enable_async_offloading")

        # gradient checkpointing layer offloading
        components.label(frame, 2, 0, "Offload Activations",
                         tooltip="Enables Activation Offloading")
        components.switch(frame, 2, 1, self.ui_state, "enable_activation_offloading")

        # gradient checkpointing layer offloading
        components.label(frame, 3, 0, "Layer offload fraction",
                         tooltip="Enables offloading of individual layers during training to reduce VRAM usage. Increases training time and uses more RAM. Only available if checkpointing is set to CPU_OFFLOADED. values between 0 and 1, 0=disabled")
        components.entry(frame, 3, 1, self.ui_state, "layer_offload_fraction")

        frame.pack(fill="both", expand=1)
        return frame

    def __ok(self):
        self.destroy()
