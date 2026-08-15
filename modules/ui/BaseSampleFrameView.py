from modules.util.enum.NoiseScheduler import NoiseScheduler
from modules.util.enum.SamplingMethod import SamplingMethod


class BaseSampleFrameView:
    def __init__(self, components):
        self.components = components

    def build_content(self, top_frame, bottom_frame, ui_state, controller, include_prompt, include_settings):
        is_flow_matching = controller.is_flow_matching()
        is_inpainting_model = controller.is_inpainting_model()
        is_video_model = controller.is_video_model()
        is_audio_model = controller.is_audio_model()
        if include_prompt:
            # prompt
            self.components.label(top_frame, 0, 0, "prompt:")
            self.components.entry(top_frame, 0, 1, ui_state, "prompt")

            # negative prompt
            if controller.supports_negative_prompt():
                self.components.label(top_frame, 1, 0, "negative prompt:")
                self.components.entry(top_frame, 1, 1, ui_state, "negative_prompt")

        if include_settings:
            # width
            self.components.label(bottom_frame, 0, 0, "width:")
            self.components.entry(bottom_frame, 0, 1, ui_state, "width")

            # height
            self.components.label(bottom_frame, 0, 2, "height:")
            self.components.entry(bottom_frame, 0, 3, ui_state, "height")

            if is_video_model:
                # frames
                self.components.label(bottom_frame, 1, 0, "frames:",
                                      tooltip="Number of frames to generate. Only used when generating videos.")
                self.components.entry(bottom_frame, 1, 1, ui_state, "frames")

            if is_audio_model:
                # length
                self.components.label(bottom_frame, 1, 2, "length:",
                                      tooltip="Length in seconds of audio output.")
                self.components.entry(bottom_frame, 1, 3, ui_state, "length")

            # seed
            self.components.label(bottom_frame, 2, 0, "seed:")
            self.components.entry(bottom_frame, 2, 1, ui_state, "seed")

            # random seed
            self.components.label(bottom_frame, 2, 2, "random seed:")
            self.components.switch(bottom_frame, 2, 3, ui_state, "random_seed")

            # cfg scale
            self.components.label(bottom_frame, 3, 0, "cfg scale:")
            self.components.entry(bottom_frame, 3, 1, ui_state, "cfg_scale")

            # sampler
            if not is_flow_matching:
                self.components.label(bottom_frame, 4, 2, "sampler:")
                self.components.options_kv(bottom_frame, 4, 3, [
                    ("DDIM", NoiseScheduler.DDIM),
                    ("Euler", NoiseScheduler.EULER),
                    ("Euler A", NoiseScheduler.EULER_A),
                    # ("DPM++", NoiseScheduler.DPMPP), # TODO: produces noisy samples
                    # ("DPM++ SDE", NoiseScheduler.DPMPP_SDE), # TODO: produces noisy samples
                    ("UniPC", NoiseScheduler.UNIPC),
                    ("Euler Karras", NoiseScheduler.EULER_KARRAS),
                    ("DPM++ Karras", NoiseScheduler.DPMPP_KARRAS),
                    ("DPM++ SDE Karras", NoiseScheduler.DPMPP_SDE_KARRAS),
                    ("UniPC Karras", NoiseScheduler.UNIPC_KARRAS)
                ], ui_state, "noise_scheduler")

            # steps
            self.components.label(bottom_frame, 4, 0, "steps:")
            self.components.entry(bottom_frame, 4, 1, ui_state, "diffusion_steps")

            if controller.model_type.has_dynamic_timestep_shift():
                self.components.label(bottom_frame, 4, 2, "override shift:",
                                      tooltip="Timestep shift for this sample, as the multiplicative factor "
                                              "(not mu). Empty uses the shift the model derives from the "
                                              "token count.")
                self.components.entry(bottom_frame, 4, 3, ui_state, "override_shift")

            if controller.supports_multiple_sampling_methods():
                self.components.label(bottom_frame, 6, 0, "sampling method:",
                                      tooltip="How this sample is generated. 'base model' walks the whole "
                                              "schedule with the transformer being trained. 'handoff to "
                                              "low-noise expert' hands over to the distilled low-noise expert "
                                              "below the expert's first sigma, as the reference pipeline does. "
                                              "'distilled' runs the expert's own short schedule from noise, so "
                                              "the trained transformer never runs. The trained LoRA applies to "
                                              "the expert as well. Both expert methods fall back to 'base "
                                              "model' when no expert model is loaded.")
                self.components.options_kv(bottom_frame, 6, 1, [
                    ("base model", SamplingMethod.STANDARD),
                    ("handoff to low-noise expert", SamplingMethod.HANDOFF_LOW_NOISE),
                    ("distilled", SamplingMethod.DISTILLED),
                ], ui_state, "sampling_method")

            # inpainting
            if is_inpainting_model:
                self.components.label(bottom_frame, 5, 0, "inpainting:",
                                      tooltip="Enables inpainting sampling. Only available when sampling from an inpainting model.")
                self.components.switch(bottom_frame, 5, 1, ui_state, "sample_inpainting")

                # base image path
                self.components.label(bottom_frame, 6, 0, "base image path:",
                                      tooltip="The base image used when inpainting.")
                self.components.path_entry(bottom_frame, 6, 1, ui_state, "base_image_path",
                                           mode="file",
                                           allow_model_files=False,
                                           allow_image_files=True,
                                           )

                # mask image path
                self.components.label(bottom_frame, 6, 2, "mask image path:",
                                      tooltip="The mask used when inpainting.")
                self.components.path_entry(bottom_frame, 6, 3, ui_state, "mask_image_path",
                                           mode="file",
                                           allow_model_files=False,
                                           allow_image_files=True,
                                           )
