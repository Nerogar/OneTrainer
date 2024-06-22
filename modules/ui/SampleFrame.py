import customtkinter as ctk

from modules.util.enum.NoiseScheduler import NoiseScheduler
from modules.util.config.SampleConfig import SampleConfig
from modules.util.ui import components
from modules.util.ui.UIState import UIState


class SampleFrame(ctk.CTkFrame):
    def __init__(
            self,
            parent,
            sample: SampleConfig,
            ui_state: UIState,
            include_prompt: bool = True,
            include_settings: bool = True,
    ):
        ctk.CTkFrame.__init__(self, parent, fg_color="transparent")

        self.sample = sample
        self.ui_state = ui_state

        if include_prompt and include_prompt:
            self.grid_rowconfigure(0, weight=0)
            self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        if include_prompt:
            top_frame = ctk.CTkFrame(self, fg_color="transparent")
            top_frame.grid(row=0, column=0, padx=0, pady=0, sticky="nsew")

            top_frame.grid_columnconfigure(0, weight=0)
            top_frame.grid_columnconfigure(1, weight=1)

        if include_settings:
            bottom_frame = ctk.CTkFrame(self, fg_color="transparent")
            bottom_frame.grid(row=1, column=0, padx=0, pady=0, sticky="nsew")

            bottom_frame.grid_columnconfigure(0, weight=0)
            bottom_frame.grid_columnconfigure(1, weight=1)
            bottom_frame.grid_columnconfigure(2, weight=0)
            bottom_frame.grid_columnconfigure(3, weight=1)

        if include_prompt:
            # prompt
            components.label(top_frame, 0, 0, "prompt:")
            components.entry(top_frame, 0, 1, self.ui_state, "prompt")

            # negative prompt
            components.label(top_frame, 1, 0, "negative prompt:")
            components.entry(top_frame, 1, 1, self.ui_state, "negative_prompt")

        if include_settings:
            # width
            components.label(bottom_frame, 0, 0, "width:")
            components.entry(bottom_frame, 0, 1, self.ui_state, "width")

            # height
            components.label(bottom_frame, 0, 2, "height:")
            components.entry(bottom_frame, 0, 3, self.ui_state, "height")

            # seed
            components.label(bottom_frame, 1, 0, "seed:")
            components.entry(bottom_frame, 1, 1, self.ui_state, "seed")

            # random seed
            components.label(bottom_frame, 1, 2, "random seed:")
            components.switch(bottom_frame, 1, 3, self.ui_state, "random_seed")

            # cfg scale
            components.label(bottom_frame, 2, 0, "cfg scale:")
            components.entry(bottom_frame, 2, 1, self.ui_state, "cfg_scale")

            # sampler
            components.label(bottom_frame, 2, 2, "sampler:")
            components.options_kv(bottom_frame, 2, 3, [
                ("DDIM", NoiseScheduler.DDIM),
                ("Euler", NoiseScheduler.EULER),
                ("Euler A", NoiseScheduler.EULER_A),
                # ("DPM++", NoiseScheduler.DPMPP), # TODO: produces noisy samples
                # ("DPM++ SDE", NoiseScheduler.DPMPP_SDE), # TODO: produces noisy samples
                ("UniPC", NoiseScheduler.UNIPC),
                ("Euler Karras", NoiseScheduler.EULER_KARRAS),
                ("DPM++ Karras", NoiseScheduler.DPMPP_KARRAS),
                ("DPM++ SDE Karras", NoiseScheduler.DPMPP_SDE_KARRAS),
                # ("UniPC Karras", NoiseScheduler.UNIPC_KARRAS),# TODO: update diffusers to fix UNIPC_KARRAS (see https://github.com/huggingface/diffusers/pull/4581)
            ], self.ui_state, "noise_scheduler")

            # steps
            components.label(bottom_frame, 3, 0, "steps:")
            components.entry(bottom_frame, 3, 1, self.ui_state, "diffusion_steps")

            # inpainting
            components.label(bottom_frame, 4, 0, "inpainting:",
                             tooltip="Enables inpainting sampling. Only available when sampling from an inpainting model.")
            components.switch(bottom_frame, 4, 1, self.ui_state, "sample_inpainting")

            # base image path
            components.label(bottom_frame, 5, 0, "base image path:",
                             tooltip="The base image used when inpainting.")
            components.file_entry(bottom_frame, 5, 1, self.ui_state, "base_image_path",
                                  allow_model_files=False,
                                  allow_image_files=True,
                                  )

            # mask image path
            components.label(bottom_frame, 5, 2, "mask image path:",
                             tooltip="The mask used when inpainting.")
            components.file_entry(bottom_frame, 5, 3, self.ui_state, "mask_image_path",
                                  allow_model_files=False,
                                  allow_image_files=True,
                                  )
