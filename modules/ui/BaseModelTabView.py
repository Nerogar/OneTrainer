
from abc import ABC, abstractmethod

from modules.util import path_util
from modules.util.enum.ConfigPart import ConfigPart
from modules.util.enum.DataType import DataType
from modules.util.enum.PathIOType import PathIOType


class BaseModelTabView(ABC):
    def __init__(self, components):
        self.components = components

    @abstractmethod
    def _make_svd_frames(self, parent, row: int):
        # Create and place SVDQuant label+entry subframes; return (label_frame, entry_frame).
        pass

    def build_content(self, frame, controller, ui_state):
        if controller.train_config.model_type.is_stable_diffusion():  # TODO simplify
            self.__setup_stable_diffusion_ui(frame, controller, ui_state)
        if controller.train_config.model_type.is_stable_diffusion_3():
            self.__setup_stable_diffusion_3_ui(frame, controller, ui_state)
        elif controller.train_config.model_type.is_stable_diffusion_xl():
            self.__setup_stable_diffusion_xl_ui(frame, controller, ui_state)
        elif controller.train_config.model_type.is_wuerstchen():
            self.__setup_wuerstchen_ui(frame, controller, ui_state)
        elif controller.train_config.model_type.is_pixart():
            self.__setup_pixart_alpha_ui(frame, controller, ui_state)
        elif controller.train_config.model_type.is_flux_1():
            self.__setup_flux_ui(frame, controller, ui_state)
        elif controller.train_config.model_type.is_flux_2():
            self.__setup_flux_2_ui(frame, controller, ui_state)
        elif controller.train_config.model_type.is_z_image():
            self.__setup_z_image_ui(frame, controller, ui_state)
        elif controller.train_config.model_type.is_chroma():
            self.__setup_chroma_ui(frame, controller, ui_state)
        elif controller.train_config.model_type.is_qwen():
            self.__setup_qwen_ui(frame, controller, ui_state)
        elif controller.train_config.model_type.is_krea2():
            self.__setup_krea2_ui(frame, controller, ui_state)
        elif controller.train_config.model_type.is_sana():
            self.__setup_sana_ui(frame, controller, ui_state)
        elif controller.train_config.model_type.is_hunyuan_video():
            self.__setup_hunyuan_video_ui(frame, controller, ui_state)
        elif controller.train_config.model_type.is_hi_dream():
            self.__setup_hi_dream_ui(frame, controller, ui_state)
        elif controller.train_config.model_type.is_ernie():
            self.__setup_ernie_ui(frame, controller, ui_state)

    def __setup_stable_diffusion_ui(self, frame, controller, ui_state):
        row = 0
        row = self.__create_base_dtype_components(frame, row, ui_state)
        row = self.__create_base_components(
            frame,
            row,
            controller,
            ui_state,
            has_unet=True,
            has_text_encoder=True,
            has_vae=True,
        )
        row = self.__create_output_components(
            frame,
            row,
            controller,
            ui_state,
        )

    def __setup_stable_diffusion_3_ui(self, frame, controller, ui_state):
        row = 0
        row = self.__create_base_dtype_components(frame, row, ui_state)
        row = self.__create_base_components(
            frame,
            row,
            controller,
            ui_state,
            has_transformer=True,
            has_text_encoder_1=True,
            has_text_encoder_2=True,
            has_text_encoder_3=True,
            has_vae=True,
        )
        row = self.__create_output_components(
            frame,
            row,
            controller,
            ui_state,
        )

    def __setup_flux_ui(self, frame, controller, ui_state):
        row = 0
        row = self.__create_base_dtype_components(frame, row, ui_state)
        row = self.__create_base_components(
            frame,
            row,
            controller,
            ui_state,
            has_transformer=True,
            allow_override_transformer=True,
            has_text_encoder_1=True,
            has_text_encoder_2=True,
            has_vae=True,
        )
        row = self.__create_output_components(
            frame,
            row,
            controller,
            ui_state,
        )

    def __setup_flux_2_ui(self, frame, controller, ui_state):
        row = 0
        row = self.__create_base_dtype_components(frame, row, ui_state)
        row = self.__create_base_components(
            frame,
            row,
            controller,
            ui_state,
            has_transformer=True,
            allow_override_transformer=True,
            has_text_encoder_1=True,
            has_vae=True,
        )
        row = self.__create_output_components(
            frame,
            row,
            controller,
            ui_state,
        )

    def __setup_z_image_ui(self, frame, controller, ui_state):
        row = 0
        row = self.__create_base_dtype_components(frame, row, ui_state)
        row = self.__create_base_components(
            frame,
            row,
            controller,
            ui_state,
            has_transformer=True,
            allow_override_transformer=True,
            has_text_encoder_1=True,
            has_vae=True,
        )
        row = self.__create_output_components(
            frame,
            row,
            controller,
            ui_state,
        )

    def __setup_ernie_ui(self, frame, controller, ui_state):
        row = 0
        row = self.__create_base_dtype_components(frame, row, ui_state)
        row = self.__create_base_components(
            frame,
            row,
            controller,
            ui_state,
            has_transformer=True,
            allow_override_transformer=True,
            has_text_encoder_1=True,
            has_vae=True,
        )
        row = self.__create_output_components(
            frame,
            row,
            controller,
            ui_state,
        )

    def __setup_chroma_ui(self, frame, controller, ui_state):
        row = 0
        row = self.__create_base_dtype_components(frame, row, ui_state)
        row = self.__create_base_components(
            frame,
            row,
            controller,
            ui_state,
            has_transformer=True,
            allow_override_transformer=True,
            has_text_encoder_1=True,
            has_vae=True,
        )
        row = self.__create_output_components(
            frame,
            row,
            controller,
            ui_state,
        )

    def __setup_qwen_ui(self, frame, controller, ui_state):
        row = 0
        row = self.__create_base_dtype_components(frame, row, ui_state)
        row = self.__create_base_components(
            frame,
            row,
            controller,
            ui_state,
            has_transformer=True,
            allow_override_transformer=True,
            has_text_encoder_1=True,
            has_vae=True,
        )
        row = self.__create_output_components(
            frame,
            row,
            controller,
            ui_state,
        )

    def __setup_krea2_ui(self, frame, controller, ui_state):
        row = 0
        row = self.__create_base_dtype_components(frame, row, ui_state)
        row = self.__create_base_components(
            frame,
            row,
            controller,
            ui_state,
            has_transformer=True,
            allow_override_transformer=True,
            has_text_encoder_1=True,
            has_vae=True,
        )
        row = self.__create_output_components(
            frame,
            row,
            controller,
            ui_state,
        )

    def __setup_stable_diffusion_xl_ui(self, frame, controller, ui_state):
        row = 0
        row = self.__create_base_dtype_components(frame, row, ui_state)
        row = self.__create_base_components(
            frame,
            row,
            controller,
            ui_state,
            has_unet=True,
            has_text_encoder_1=True,
            has_text_encoder_2=True,
            has_vae=True,
        )
        row = self.__create_output_components(
            frame,
            row,
            controller,
            ui_state,
        )

    def __setup_wuerstchen_ui(self, frame, controller, ui_state):
        row = 0
        row = self.__create_base_dtype_components(frame, row, ui_state)
        row = self.__create_base_components(
            frame,
            row,
            controller,
            ui_state,
            has_prior=True,
            allow_override_prior=controller.train_config.model_type.is_stable_cascade(),
            has_text_encoder=True,
        )
        row = self.__create_effnet_encoder_components(frame, row, ui_state)
        row = self.__create_decoder_components(frame, row, ui_state, controller.train_config.model_type.is_wuerstchen_v2())
        row = self.__create_output_components(
            frame,
            row,
            controller,
            ui_state,
        )

    def __setup_pixart_alpha_ui(self, frame, controller, ui_state):
        row = 0
        row = self.__create_base_dtype_components(frame, row, ui_state)
        row = self.__create_base_components(
            frame,
            row,
            controller,
            ui_state,
            has_transformer=True,
            has_text_encoder=True,
            has_vae=True,
        )
        row = self.__create_output_components(
            frame,
            row,
            controller,
            ui_state,
        )

    def __setup_sana_ui(self, frame, controller, ui_state):
        row = 0
        row = self.__create_base_dtype_components(frame, row, ui_state)
        row = self.__create_base_components(
            frame,
            row,
            controller,
            ui_state,
            has_transformer=True,
            has_text_encoder=True,
            has_vae=True,
        )
        row = self.__create_output_components(
            frame,
            row,
            controller,
            ui_state,
        )

    def __setup_hunyuan_video_ui(self, frame, controller, ui_state):
        row = 0
        row = self.__create_base_dtype_components(frame, row, ui_state)
        row = self.__create_base_components(
            frame,
            row,
            controller,
            ui_state,
            has_transformer=True,
            allow_override_transformer=True,
            has_text_encoder_1=True,
            has_text_encoder_2=True,
            has_vae=True,
        )
        row = self.__create_output_components(
            frame,
            row,
            controller,
            ui_state,
        )

    def __setup_hi_dream_ui(self, frame, controller, ui_state):
        row = 0
        row = self.__create_base_dtype_components(frame, row, ui_state)
        row = self.__create_base_components(
            frame,
            row,
            controller,
            ui_state,
            has_transformer=True,
            has_text_encoder_1=True,
            has_text_encoder_2=True,
            has_text_encoder_3=True,
            has_text_encoder_4=True,
            allow_override_text_encoder_4=True,
            has_vae=True,
        )
        row = self.__create_output_components(
            frame,
            row,
            controller,
            ui_state,
        )

    def __create_dtype_options(self, include_gguf: bool = False, include_a8: bool = False) -> list[tuple[str, DataType]]:
        options = [
            ("float32", DataType.FLOAT_32),
            ("bfloat16", DataType.BFLOAT_16),
            ("float16", DataType.FLOAT_16),
            ("float8 (W8)", DataType.FLOAT_8),
            # ("int8", DataType.INT_8),  # TODO: reactivate when the int8 implementation is fixed in bitsandbytes: https://github.com/bitsandbytes-foundation/bitsandbytes/issues/1332
            ("nfloat4", DataType.NFLOAT_4),
        ]
        if include_a8:
            options += [
                ("float W8A8", DataType.FLOAT_W8A8),
                ("int W8A8", DataType.INT_W8A8),
            ]

        if include_gguf:
            options.append(("GGUF", DataType.GGUF))
            if include_a8:
                options += [
                    ("GGUF A8 float", DataType.GGUF_A8_FLOAT),
                    ("GGUF A8 int", DataType.GGUF_A8_INT),
                ]

        return options

    def __create_base_dtype_components(self, frame, row: int, ui_state) -> int:
        # huggingface token
        self.components.label(frame, row, 0, "Hugging Face Token",
                         tooltip="Enter your Hugging Face access token if you have used a protected Hugging Face repository below.\nThis value is stored separately, not saved to your configuration file. "
                                 "Go to https://huggingface.co/settings/tokens to create an access token.",
                         wide_tooltip=True)
        self.components.entry(frame, row, 1, ui_state, "secrets.huggingface_token")

        row += 1

        # base model
        self.components.label(frame, row, 0, "Base Model",
                         tooltip="Filename, directory or Hugging Face repository of the base model")
        self.components.path_entry(
            frame, row, 1, ui_state, "base_model_name",
            mode="file", path_modifier=path_util.json_path_modifier
        )

        # compile
        self.components.label(frame, row, 3, "Compile transformer blocks",
                         tooltip="Uses torch.compile and Triton to significantly speed up training. Only applies to transformer/unet. Disable in case of compatibility issues.")
        self.components.switch(frame, row, 4, ui_state, "compile")

        row += 1

        return row

    def __create_base_components(
            self,
            frame,
            row: int,
            controller,
            ui_state,
            has_unet: bool = False,
            has_prior: bool = False,
            allow_override_prior: bool = False,
            has_transformer: bool = False,
            allow_override_transformer: bool = False,
            allow_override_text_encoder_4: bool = False,
            has_text_encoder: bool = False,
            has_text_encoder_1: bool = False,
            has_text_encoder_2: bool = False,
            has_text_encoder_3: bool = False,
            has_text_encoder_4: bool = False,
            has_vae: bool = False,
    ) -> int:
        if has_unet:
            # unet weight dtype
            self.components.label(frame, row, 3, "UNet Data Type",
                             tooltip="The unet weight data type")
            self.components.options_kv(frame, row, 4, self.__create_dtype_options(include_a8=True),
                                  ui_state, "unet.weight_dtype")

            row += 1

        if has_prior:
            if allow_override_prior:
                # prior model
                self.components.label(frame, row, 0, "Prior Model",
                                 tooltip="Filename, directory or Hugging Face repository of the prior model")
                self.components.path_entry(
                    frame, row, 1, ui_state, "prior.model_name",
                    mode="file", path_modifier=path_util.json_path_modifier
                )

            # prior weight dtype
            self.components.label(frame, row, 3, "Prior Data Type",
                             tooltip="The prior weight data type")
            self.components.options_kv(frame, row, 4, self.__create_dtype_options(),
                                  ui_state, "prior.weight_dtype")

            row += 1

        if has_transformer:
            if allow_override_transformer:
                # transformer model
                self.components.label(frame, row, 0, "Override Transformer / GGUF",
                                 tooltip="Can be used to override the transformer in the base model. Safetensors and GGUF files are supported, local and on Huggingface. If a GGUF file is used, the DataType must also be set to GGUF")
                self.components.path_entry(
                    frame, row, 1, ui_state, "transformer.model_name",
                    mode="file", path_modifier=path_util.json_path_modifier
                )

            # transformer weight dtype
            self.components.label(frame, row, 3, "Transformer Data Type",
                             tooltip="The transformer weight data type")
            self.components.options_kv(frame, row, 4, self.__create_dtype_options(include_gguf=True, include_a8=True),
                                  ui_state, "transformer.weight_dtype")

            row += 1

        presets = controller.get_presets()

        self.components.label(frame, row, 0, "Quantization")
        self.components.layer_filter_entry(frame, row, 1, ui_state,
            preset_var_name="quantization.layer_filter_preset", presets=presets,
            preset_label="Quantization Layer Filter",
            preset_tooltip="Select a preset defining which layers to quantize. Quantization of certain layers can decrease model quality. Only applies to the transformer/unet",
            entry_var_name="quantization.layer_filter",
            entry_tooltip="Comma-separated list of layers to quantize. Regular expressions (if toggled) are supported. Any model layer with a matching name will be quantized",
            regex_var_name="quantization.layer_filter_regex",
            regex_tooltip="If enabled, layer filter patterns are interpreted as regular expressions. Otherwise, simple substring matching is used.",
            frame_color="transparent",
        )

        # SVDQuant - create vertical grids to match the size of layer_filter_entry
        svd_label_frame, svd_entry_frame = self._make_svd_frames(frame, row)
        self.components.label(svd_label_frame, 0, 0, "SVDQuant",
                         tooltip="What datatype to use for SVDQuant weights decomposition.")
        self.components.options_kv(svd_entry_frame, 0, 0, [("disabled", DataType.NONE), ("float32", DataType.FLOAT_32), ("bfloat16", DataType.BFLOAT_16)],
                              ui_state, "quantization.svd_dtype")
        self.components.label(svd_label_frame, 1, 0, "SVDQuant Rank",
                         tooltip="Rank for SVDQuant weights decomposition")
        self.components.entry(svd_entry_frame, 1, 0, ui_state, "quantization.svd_rank")
        row += 1

        if has_text_encoder:
            # text encoder weight dtype
            self.components.label(frame, row, 3, "Text Encoder Data Type",
                             tooltip="The text encoder weight data type")
            self.components.options_kv(frame, row, 4, self.__create_dtype_options(),
                                  ui_state, "text_encoder.weight_dtype")

            row += 1

        if has_text_encoder_1:
            # text encoder 1 weight dtype
            self.components.label(frame, row, 3, "Text Encoder 1 Data Type",
                             tooltip="The text encoder 1 weight data type")
            self.components.options_kv(frame, row, 4, self.__create_dtype_options(),
                                  ui_state, "text_encoder.weight_dtype")

            row += 1

        if has_text_encoder_2:
            # text encoder 2 weight dtype
            self.components.label(frame, row, 3, "Text Encoder 2 Data Type",
                             tooltip="The text encoder 2 weight data type")
            self.components.options_kv(frame, row, 4, self.__create_dtype_options(),
                                  ui_state, "text_encoder_2.weight_dtype")

            row += 1

        if has_text_encoder_3:
            # text encoder 3 weight dtype
            self.components.label(frame, row, 3, "Text Encoder 3 Data Type",
                             tooltip="The text encoder 3 weight data type")
            self.components.options_kv(frame, row, 4, self.__create_dtype_options(),
                                  ui_state, "text_encoder_3.weight_dtype")

            row += 1

        if has_text_encoder_4:
            if allow_override_text_encoder_4:
                # text encoder 4 weight dtype
                self.components.label(frame, row, 0, "Text Encoder 4 Override",
                                 tooltip="Filename, directory or Hugging Face repository of the text encoder 4 model")
                self.components.path_entry(
                    frame, row, 1, ui_state, "text_encoder_4.model_name",
                    mode="file", path_modifier=path_util.json_path_modifier
                )

            # text encoder 4 weight dtype
            self.components.label(frame, row, 3, "Text Encoder 4 Data Type",
                             tooltip="The text encoder 4 weight data type")
            self.components.options_kv(frame, row, 4, self.__create_dtype_options(),
                                  ui_state, "text_encoder_4.weight_dtype")

            row += 1

        if has_vae:
            # base model
            self.components.label(frame, row, 0, "VAE Override",
                             tooltip="Directory or Hugging Face repository of a VAE model in diffusers format. Can be used to override the VAE included in the base model. Using a safetensor VAE file will cause an error that the model cannot be loaded.")
            self.components.path_entry(
                frame, row, 1, ui_state, "vae.model_name",
                mode="file", path_modifier=path_util.json_path_modifier
            )

            # vae weight dtype
            self.components.label(frame, row, 3, "VAE Data Type",
                             tooltip="The vae weight data type")
            self.components.options_kv(frame, row, 4, self.__create_dtype_options(),
                                  ui_state, "vae.weight_dtype")

            row += 1

        return row

    def __create_effnet_encoder_components(self, frame, row: int, ui_state) -> int:
        # effnet encoder model
        self.components.label(frame, row, 0, "Effnet Encoder Model",
                         tooltip="Filename, directory or Hugging Face repository of the effnet encoder model")
        self.components.path_entry(
            frame, row, 1, ui_state, "effnet_encoder.model_name",
            mode="file", path_modifier=path_util.json_path_modifier
        )

        # effnet encoder weight dtype
        self.components.label(frame, row, 3, "Effnet Encoder Data Type",
                         tooltip="The effnet encoder weight data type")
        self.components.options_kv(frame, row, 4, self.__create_dtype_options(),
                              ui_state, "effnet_encoder.weight_dtype")

        row += 1

        return row

    def __create_decoder_components(
            self,
            frame,
            row: int,
            ui_state,
            has_text_encoder: bool,
    ) -> int:
        # decoder model
        self.components.label(frame, row, 0, "Decoder Model",
                         tooltip="Filename, directory or Hugging Face repository of the decoder model")
        self.components.path_entry(
            frame, row, 1, ui_state, "decoder.model_name",
            mode="file", path_modifier=path_util.json_path_modifier
        )

        # decoder weight dtype
        self.components.label(frame, row, 3, "Decoder Data Type",
                         tooltip="The decoder weight data type")
        self.components.options_kv(frame, row, 4, self.__create_dtype_options(),
                              ui_state, "decoder.weight_dtype")

        row += 1

        if has_text_encoder:
            # decoder text encoder weight dtype
            self.components.label(frame, row, 3, "Decoder Text Encoder Data Type",
                             tooltip="The decoder text encoder weight data type")
            self.components.options_kv(frame, row, 4, self.__create_dtype_options(),
                                  ui_state, "decoder_text_encoder.weight_dtype")

            row += 1

        # decoder vqgan weight dtype
        self.components.label(frame, row, 3, "Decoder VQGAN Data Type",
                         tooltip="The decoder vqgan weight data type")
        self.components.options_kv(frame, row, 4, self.__create_dtype_options(),
                              ui_state, "decoder_vqgan.weight_dtype")

        row += 1

        return row

    def __create_output_components(
            self,
            frame,
            row: int,
            controller,
            ui_state,
    ) -> int:
        # output model destination
        self.components.label(frame, row, 0, "Model Output Destination",
                         tooltip="Filename or directory where the output model is saved")
        self.components.path_entry(
            frame, row, 1, ui_state, "output_model_destination",
            mode="file",
            io_type=PathIOType.MODEL,
        )

        # output data type
        self.components.label(frame, row, 3, "Output Data Type",
                         tooltip="Precision to use when saving the output model")
        self.components.options_kv(frame, row, 4, [
            ("float16", DataType.FLOAT_16),
            ("float32", DataType.FLOAT_32),
            ("bfloat16", DataType.BFLOAT_16),
            ("float8", DataType.FLOAT_8),
            ("nfloat4", DataType.NFLOAT_4),
        ], ui_state, "output_dtype")

        row += 1

        # output format
        formats = controller.get_output_formats()

        self.components.label(frame, row, 0, "Output Format",
                         tooltip="Format to use when saving the output model")
        self.components.options_kv(frame, row, 1, formats, ui_state, "output_model_format")

        # include config
        self.components.label(frame, row, 3, "Include Config",
                         tooltip="Include the training configuration in the final model. Only supported for safetensors files. "
                                 "None: No config is included. "
                                 "Settings: All training settings are included. "
                                 "All: All settings, including the samples and concepts are included.")
        self.components.options_kv(frame, row, 4, [
            ("None", ConfigPart.NONE),
            ("Settings", ConfigPart.SETTINGS),
            ("All", ConfigPart.ALL),
        ], ui_state, "include_train_config")

        row += 1

        return row
