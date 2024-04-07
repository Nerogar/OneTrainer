from pathlib import Path

import customtkinter as ctk

from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ConfigPart import ConfigPart
from modules.util.enum.DataType import DataType
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.ui import components
from modules.util.ui.UIState import UIState


class ModelTab:

    def __init__(self, master, train_config: TrainConfig, ui_state: UIState):
        super(ModelTab, self).__init__()

        self.master = master
        self.train_config = train_config
        self.ui_state = ui_state

        master.grid_rowconfigure(0, weight=1)
        master.grid_columnconfigure(0, weight=1)

        self.scroll_frame = None

        self.refresh_ui()

    def refresh_ui(self):
        if self.scroll_frame:
            self.scroll_frame.destroy()

        self.scroll_frame = ctk.CTkScrollableFrame(self.master, fg_color="transparent")
        self.scroll_frame.grid(row=0, column=0, sticky="nsew")

        self.scroll_frame.grid_columnconfigure(0, weight=0)
        self.scroll_frame.grid_columnconfigure(1, weight=10)
        self.scroll_frame.grid_columnconfigure(2, minsize=50)
        self.scroll_frame.grid_columnconfigure(3, weight=0)
        self.scroll_frame.grid_columnconfigure(4, weight=1)

        if self.train_config.model_type.is_stable_diffusion():
            self.__setup_stable_diffusion_ui()
        elif self.train_config.model_type.is_stable_diffusion_xl():
            self.__setup_stable_diffusion_xl_ui()
        elif self.train_config.model_type.is_wuerstchen():
            self.__setup_wuerstchen_ui()
        elif self.train_config.model_type.is_pixart_alpha():
            self.__setup_pixart_alpha_ui()

    def __setup_stable_diffusion_ui(self):
        row = 0
        row = self.__create_base_dtype_components(row)
        row = self.__create_base_components(
            row,
            has_unet=True,
            has_text_encoder=True,
            has_vae=True,
        )
        row = self.__create_output_components(
            row,
            allow_safetensors=True,
            allow_diffusers=self.train_config.training_method in [
                TrainingMethod.FINE_TUNE,
                TrainingMethod.FINE_TUNE_VAE,
            ],
            allow_checkpoint=True,
        )

    def __setup_stable_diffusion_xl_ui(self):
        row = 0
        row = self.__create_base_dtype_components(row)
        row = self.__create_base_components(
            row,
            has_unet=True,
            has_text_encoder_1=True,
            has_text_encoder_2=True,
            has_vae=True,
        )
        row = self.__create_output_components(
            row,
            allow_safetensors=True,
            allow_diffusers=self.train_config.training_method == TrainingMethod.FINE_TUNE,
            allow_checkpoint=True,
        )

    def __setup_wuerstchen_ui(self):
        row = 0
        row = self.__create_base_dtype_components(row)
        row = self.__create_base_components(
            row,
            has_prior=True,
            allow_override_prior=self.train_config.model_type.is_stable_cascade(),
            has_text_encoder=True,
        )
        row = self.__create_effnet_encoder_components(row)
        row = self.__create_decoder_components(row, self.train_config.model_type.is_wuerstchen_v2())
        row = self.__create_output_components(
            row,
            allow_safetensors=self.train_config.training_method != TrainingMethod.FINE_TUNE
                              or self.train_config.model_type.is_stable_cascade(),
            allow_diffusers=self.train_config.training_method == TrainingMethod.FINE_TUNE,
            allow_checkpoint=self.train_config.training_method != TrainingMethod.FINE_TUNE,
        )

    def __setup_pixart_alpha_ui(self):
        row = 0
        row = self.__create_base_dtype_components(row)
        row = self.__create_base_components(
            row,
            has_prior=True,
            has_text_encoder=True,
            has_vae=True,
        )
        row = self.__create_output_components(
            row,
            allow_safetensors=True,
            allow_diffusers=self.train_config.training_method == TrainingMethod.FINE_TUNE,
            allow_checkpoint=True,
        )

    def __create_base_dtype_components(self, row: int) -> int:
        # base model
        components.label(self.scroll_frame, row, 0, "Base Model",
                         tooltip="Filename, directory or Hugging Face repository of the base model")
        components.file_entry(
            self.scroll_frame, row, 1, self.ui_state, "base_model_name",
            path_modifier=lambda x: Path(x).parent.absolute() if x.endswith(".json") else x
        )

        # weight dtype
        components.label(self.scroll_frame, row, 3, "Weight Data Type",
                         tooltip="The base model weight data type used for training. This can reduce memory consumption, but reduces precision")
        components.options_kv(self.scroll_frame, row, 4, [
            ("float32", DataType.FLOAT_32),
            ("bfloat16", DataType.BFLOAT_16),
            ("float16", DataType.FLOAT_16),
            ("float8", DataType.FLOAT_8),
        ], self.ui_state, "weight_dtype")

        row += 1

        return row

    def __create_base_components(
            self,
            row: int,
            has_unet: bool = False,
            has_prior: bool = False,
            allow_override_prior: bool = False,
            has_text_encoder: bool = False,
            has_text_encoder_1: bool = False,
            has_text_encoder_2: bool = False,
            has_vae: bool = False,
    ) -> int:
        if has_unet:
            # unet weight dtype
            components.label(self.scroll_frame, row, 3, "Override UNet Data Type",
                             tooltip="Overrides the unet weight data type")
            components.options_kv(self.scroll_frame, row, 4, [
                ("", DataType.NONE),
                ("float32", DataType.FLOAT_32),
                ("bfloat16", DataType.BFLOAT_16),
                ("float16", DataType.FLOAT_16),
                ("float8", DataType.FLOAT_8),
            ], self.ui_state, "unet.weight_dtype")

            row += 1

        if has_prior:
            if allow_override_prior:
                # prior model
                components.label(self.scroll_frame, row, 0, "Prior Model",
                                 tooltip="Filename, directory or Hugging Face repository of the prior model")
                components.file_entry(
                    self.scroll_frame, row, 1, self.ui_state, "prior.model_name",
                    path_modifier=lambda x: Path(x).parent.absolute() if x.endswith(".json") else x
                )

            # prior weight dtype
            components.label(self.scroll_frame, row, 3, "Override Prior Data Type",
                             tooltip="Overrides the prior weight data type")
            components.options_kv(self.scroll_frame, row, 4, [
                ("", DataType.NONE),
                ("float32", DataType.FLOAT_32),
                ("bfloat16", DataType.BFLOAT_16),
                ("float16", DataType.FLOAT_16),
                ("float8", DataType.FLOAT_8),
            ], self.ui_state, "prior.weight_dtype")

            row += 1

        if has_text_encoder:
            # text encoder weight dtype
            components.label(self.scroll_frame, row, 3, "Override Text Encoder Data Type",
                             tooltip="Overrides the text encoder weight data type")
            components.options_kv(self.scroll_frame, row, 4, [
                ("", DataType.NONE),
                ("float32", DataType.FLOAT_32),
                ("bfloat16", DataType.BFLOAT_16),
                ("float16", DataType.FLOAT_16),
                ("float8", DataType.FLOAT_8),
            ], self.ui_state, "text_encoder.weight_dtype")

            row += 1

        if has_text_encoder_1:
            # text encoder 1 weight dtype
            components.label(self.scroll_frame, row, 3, "Override Text Encoder 1 Data Type",
                             tooltip="Overrides the text encoder 1 weight data type")
            components.options_kv(self.scroll_frame, row, 4, [
                ("", DataType.NONE),
                ("float32", DataType.FLOAT_32),
                ("bfloat16", DataType.BFLOAT_16),
                ("float16", DataType.FLOAT_16),
                ("float8", DataType.FLOAT_8),
            ], self.ui_state, "text_encoder.weight_dtype")

            row += 1

        if has_text_encoder_2:
            # text encoder 2 weight dtype
            components.label(self.scroll_frame, row, 3, "Override Text Encoder 2 Data Type",
                             tooltip="Overrides the text encoder 2 weight data type")
            components.options_kv(self.scroll_frame, row, 4, [
                ("", DataType.NONE),
                ("float32", DataType.FLOAT_32),
                ("bfloat16", DataType.BFLOAT_16),
                ("float16", DataType.FLOAT_16),
                ("float8", DataType.FLOAT_8),
            ], self.ui_state, "text_encoder_2.weight_dtype")

            row += 1

        if has_vae:
            # base model
            components.label(self.scroll_frame, row, 0, "VAE Override",
                             tooltip="Directory or Hugging Face repository of a VAE model in diffusers format. Can be used to override the VAE included in the base model. Using a safetensor VAE file will cause an error that the model cannot be loaded.")
            components.file_entry(
                self.scroll_frame, row, 1, self.ui_state, "vae.model_name",
                path_modifier=lambda x: Path(x).parent.absolute() if x.endswith(".json") else x
            )

            # vae weight dtype
            components.label(self.scroll_frame, row, 3, "Override VAE Data Type",
                             tooltip="Overrides the vae weight data type")
            components.options_kv(self.scroll_frame, row, 4, [
                ("", DataType.NONE),
                ("float32", DataType.FLOAT_32),
                ("bfloat16", DataType.BFLOAT_16),
                ("float16", DataType.FLOAT_16),
                ("float8", DataType.FLOAT_8),
            ], self.ui_state, "vae.weight_dtype")

            row += 1

        return row

    def __create_effnet_encoder_components(self, row: int):
        # effnet encoder model
        components.label(self.scroll_frame, row, 0, "Effnet Encoder Model",
                         tooltip="Filename, directory or Hugging Face repository of the effnet encoder model")
        components.file_entry(
            self.scroll_frame, row, 1, self.ui_state, "effnet_encoder.model_name",
            path_modifier=lambda x: Path(x).parent.absolute() if x.endswith(".json") else x
        )

        # effnet encoder weight dtype
        components.label(self.scroll_frame, row, 3, "Override Effnet Encoder Data Type",
                         tooltip="Overrides the effnet encoder weight data type")
        components.options_kv(self.scroll_frame, row, 4, [
            ("", DataType.NONE),
            ("float32", DataType.FLOAT_32),
            ("bfloat16", DataType.BFLOAT_16),
            ("float16", DataType.FLOAT_16),
            ("float8", DataType.FLOAT_8),
        ], self.ui_state, "effnet_encoder.weight_dtype")

        row += 1

        return row

    def __create_decoder_components(
            self,
            row: int,
            has_text_encoder: bool,
    ) -> int:
        # decoder model
        components.label(self.scroll_frame, row, 0, "Decoder Model",
                         tooltip="Filename, directory or Hugging Face repository of the decoder model")
        components.file_entry(
            self.scroll_frame, row, 1, self.ui_state, "decoder.model_name",
            path_modifier=lambda x: Path(x).parent.absolute() if x.endswith(".json") else x
        )

        # decoder weight dtype
        components.label(self.scroll_frame, row, 3, "Override Decoder Data Type",
                         tooltip="Overrides the decoder weight data type")
        components.options_kv(self.scroll_frame, row, 4, [
            ("", DataType.NONE),
            ("float32", DataType.FLOAT_32),
            ("bfloat16", DataType.BFLOAT_16),
            ("float16", DataType.FLOAT_16),
            ("float8", DataType.FLOAT_8),
        ], self.ui_state, "decoder.weight_dtype")

        row += 1

        if has_text_encoder:
            # decoder text encoder weight dtype
            components.label(self.scroll_frame, row, 3, "Override Decoder Text Encoder Data Type",
                             tooltip="Overrides the decoder text encoder weight data type")
            components.options_kv(self.scroll_frame, row, 4, [
                ("", DataType.NONE),
                ("float32", DataType.FLOAT_32),
                ("bfloat16", DataType.BFLOAT_16),
                ("float16", DataType.FLOAT_16),
                ("float8", DataType.FLOAT_8),
            ], self.ui_state, "decoder_text_encoder.weight_dtype")

            row += 1

        # decoder vqgan weight dtype
        components.label(self.scroll_frame, row, 3, "Override Decoder VQGAN Data Type",
                         tooltip="Overrides the decoder vqgan weight data type")
        components.options_kv(self.scroll_frame, row, 4, [
            ("", DataType.NONE),
            ("float32", DataType.FLOAT_32),
            ("bfloat16", DataType.BFLOAT_16),
            ("float16", DataType.FLOAT_16),
            ("float8", DataType.FLOAT_8),
        ], self.ui_state, "decoder_vqgan.weight_dtype")

        row += 1

        return row

    def __create_output_components(
            self,
            row: int,
            allow_safetensors: bool = False,
            allow_diffusers: bool = False,
            allow_checkpoint: bool = False,
    ) -> int:
        # output model destination
        components.label(self.scroll_frame, row, 0, "Model Output Destination",
                         tooltip="Filename or directory where the output model is saved")
        components.file_entry(self.scroll_frame, row, 1, self.ui_state, "output_model_destination", is_output=True)

        # output data type
        components.label(self.scroll_frame, row, 3, "Output Data Type",
                         tooltip="Precision to use when saving the output model")
        components.options_kv(self.scroll_frame, row, 4, [
            ("float16", DataType.FLOAT_16),
            ("float32", DataType.FLOAT_32),
            ("bfloat16", DataType.BFLOAT_16),
            ("float8", DataType.FLOAT_8),
        ], self.ui_state, "output_dtype")

        row += 1

        # output format
        formats = []
        if allow_safetensors:
            formats.append(("Safetensors", ModelFormat.SAFETENSORS))
        if allow_diffusers:
            formats.append(("Diffusers", ModelFormat.DIFFUSERS))
        if allow_checkpoint:
            formats.append(("Checkpoint", ModelFormat.CKPT))

        components.label(self.scroll_frame, row, 0, "Output Format",
                         tooltip="Format to use when saving the output model")
        components.options_kv(self.scroll_frame, row, 1, formats, self.ui_state, "output_model_format")

        # include config
        components.label(self.scroll_frame, row, 3, "Include Config",
                         tooltip="Include the training configuration in the final model. Only supported for safetensors files. "
                                 "None: No config is included. "
                                 "Settings: All training settings are included. "
                                 "All: All settings, including the samples and concepts are included.")
        components.options_kv(self.scroll_frame, row, 4, [
            ("None", ConfigPart.NONE),
            ("Settings", ConfigPart.SETTINGS),
            ("All", ConfigPart.ALL),
        ], self.ui_state, "include_train_config")

        row += 1

        return row
