import copy

import customtkinter as ctk
from PIL import Image

from modules.ui.SampleFrame import SampleFrame
from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.params.SampleParams import SampleParams
from modules.util.ui import components
from modules.util.ui.UIState import UIState


class SampleWindow(ctk.CTkToplevel):
    def __init__(
            self,
            parent,
            callbacks: TrainCallbacks,
            commands: TrainCommands,
            *args, **kwargs
    ):
        ctk.CTkToplevel.__init__(self, parent, *args, **kwargs)

        self.callbacks = callbacks
        self.commands = commands

        self.sample = SampleParams.default_values()
        self.ui_state = UIState(self, self.sample)

        self.title("Sample")
        self.geometry("1200x800")
        self.resizable(True, True)
        self.wait_visibility()
        self.grab_set()
        self.focus_set()

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        frame = SampleFrame(self, self.sample, self.ui_state)
        frame.grid(row=0, column=0, padx=0, pady=0, sticky="nsew")

        # image
        self.image = ctk.CTkImage(
            light_image=self.__dummy_image(),
            size=(512, 512)
        )
        image_label = ctk.CTkLabel(master=self, text="", image=self.image, height=512, width=512)
        image_label.grid(row=0, column=1, rowspan=2)

        components.button(self, 1, 0, "sample", self.__sample)

    def update_preview(self, image: Image):
        self.image.configure(light_image=image)

    def __dummy_image(self) -> Image:
        return Image.new(mode="RGB", size=(512, 512), color=(0, 0, 0))

    def __sample(self):
        self.commands.sample_custom(copy.copy(self.sample))
