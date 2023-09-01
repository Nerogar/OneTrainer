import customtkinter as ctk

from modules.util.enum.NoiseScheduler import NoiseScheduler
from modules.util.params.SampleParams import SampleParams
from modules.util.ui import components
from modules.util.ui.UIState import UIState


class SampleWindow(ctk.CTkToplevel):
    def __init__(self, parent, sample: SampleParams, ui_state: UIState, *args, **kwargs):
        ctk.CTkToplevel.__init__(self, parent, *args, **kwargs)

        self.sample = sample
        self.ui_state = ui_state

        self.title("Sample")
        self.geometry("800x450")
        self.resizable(False, False)
        self.wait_visibility()
        self.grab_set()
        self.focus_set()

        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        top_frame = ctk.CTkFrame(self, fg_color="transparent")
        top_frame.grid(row=0, column=0, padx=0, pady=0, sticky="nsew")

        top_frame.grid_columnconfigure(0, weight=0)
        top_frame.grid_columnconfigure(1, weight=1)

        bottom_frame = ctk.CTkFrame(self, fg_color="transparent")
        bottom_frame.grid(row=1, column=0, padx=0, pady=0, sticky="nsew")

        bottom_frame.grid_columnconfigure(0, weight=0)
        bottom_frame.grid_columnconfigure(1, weight=1)
        bottom_frame.grid_columnconfigure(2, weight=0)
        bottom_frame.grid_columnconfigure(3, weight=1)

        # prompt
        components.label(top_frame, 0, 0, "prompt:")
        components.entry(top_frame, 0, 1, self.ui_state, "prompt")

        # negative prompt
        components.label(top_frame, 1, 0, "negative prompt:")
        components.entry(top_frame, 1, 1, self.ui_state, "negative_prompt")

        # width
        components.label(bottom_frame, 0, 0, "width:")
        components.entry(bottom_frame, 0, 1, self.ui_state, "width")

        # height
        components.label(bottom_frame, 0, 2, "height:")
        components.entry(bottom_frame, 0, 3, self.ui_state, "height")

        # steps
        components.label(bottom_frame, 1, 0, "steps:")
        components.entry(bottom_frame, 1, 1, self.ui_state, "diffusion_steps")

        # seed
        components.label(bottom_frame, 1, 2, "seed:")
        components.entry(bottom_frame, 1, 3, self.ui_state, "seed")

        # cfg scale
        components.label(bottom_frame, 2, 0, "cfg scale:")
        components.entry(bottom_frame, 2, 1, self.ui_state, "cfg_scale")

        # sampler
        components.label(bottom_frame, 2, 2, "sampler:")
        components.options_kv(bottom_frame, 2, 3, [
            ("DDIM", NoiseScheduler.DDIM),
            ("Euler", NoiseScheduler.EULER),
            ("Euler A", NoiseScheduler.EULER_A),
        ], self.ui_state, "noise_scheduler")

        components.button(self, 2, 0, "ok", self.__ok)

    def __ok(self):
        self.destroy()
