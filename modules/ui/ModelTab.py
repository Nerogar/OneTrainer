from pathlib import Path

from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ConfigPart import ConfigPart
from modules.util.enum.DataType import DataType
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.ui import components
from modules.util.ui.UIState import UIState

import customtkinter as ctk


class ModelTab:

    def __init__(self, master, train_config: TrainConfig, ui_state: UIState):
        super().__init__()

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
        if self.train_config.model_type.is_stable_diffusion_3():
            self.__setup_stable_diffusion_3_ui()
        elif self.train_config.model_type.is_stable_diffusion_xl():
            self.__setup_stable_diffusion_xl_ui()
        elif self.train_config.model_type.is_wuerstchen():
            self.__setup_wuerstchen_ui()
        elif self.train_config.model_type.is_pixart():
            self.__setup_pixart_alpha_ui()
        elif self.train_config.model_type.is_flux():
            self.__setup_flux_ui()

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

    def __setup_stable_diffusion_3_ui(self):
        row = 0
        row = self.__create_base_dtype_components(row)
        row = self.__create_base_components(
            row,
            has_prior=True,
            has_text_encoder_1=True,
            has_text_encoder_2=True,
            has_text_encoder_3=True,
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

    def __setup_flux_ui(self):
        row = 0
        row = self.__create_base_dtype_components(row)
        row = self.__create_base_components(
            row,
            has_prior=True,
            has_text_encoder_1=True,
            has_text_encoder_2=True,
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

    def __create_dtype_options(self, include_none:bool=True) -> list[tuple[str, DataType]]:
        options = [
            ("float32", DataType.FLOAT_32),
            ("bfloat16", DataType.BFLOAT_16),
            ("float16", DataType.FLOAT_16),
            ("float8", DataType.FLOAT_8),
            # ("int8", DataType.INT_8),  # TODO: reactivate when the int8 implementation is fixed in bitsandbytes: https://github.com/bitsandbytes-foundation/bitsandbytes/issues/1332
            ("nfloat4", DataType.NFLOAT_4),
        ]

        if include_none:
            options.insert(0, ("", DataType.NONE))

        return options


    def __create_base_dtype_components(self, row: int) -> int:
        # base model
        components.label(self.scroll_frame, row, 0, "基础模型",
                         tooltip="基础模型的文件名、目录或 Hugging Face 仓库")
        components.file_entry(
            self.scroll_frame, row, 1, self.ui_state, "base_model_name",
            path_modifier=lambda x: Path(x).parent.absolute() if x.endswith(".json") else x
        )

        # weight dtype
        components.label(self.scroll_frame, row, 3, "权重数据类型",
                         tooltip="用于训练的基础模型权重数据类型。这可以减少内存消耗，但会降低精度")
        components.options_kv(self.scroll_frame, row, 4, self.__create_dtype_options(False),
                              self.ui_state, "weight_dtype")

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
            has_text_encoder_3: bool = False,
            has_vae: bool = False,
    ) -> int:
        if has_unet:
            # unet weight dtype
            components.label(self.scroll_frame, row, 3, "覆盖 UNet 数据类型",
                             tooltip="覆盖 UNet 权重数据类型")
            components.options_kv(self.scroll_frame, row, 4, self.__create_dtype_options(),
                                  self.ui_state, "unet.weight_dtype")

            row += 1

        if has_prior:
            if allow_override_prior:
                # prior model
                components.label(self.scroll_frame, row, 0, "先验模型",
                                 tooltip="先验模型的文件名、目录或 Hugging Face 仓库")
                components.file_entry(
                    self.scroll_frame, row, 1, self.ui_state, "prior.model_name",
                    path_modifier=lambda x: Path(x).parent.absolute() if x.endswith(".json") else x
                )

            # prior weight dtype
            components.label(self.scroll_frame, row, 3, "覆盖先验数据类型",
                             tooltip="覆盖先验权重数据类型")
            components.options_kv(self.scroll_frame, row, 4,  self.__create_dtype_options(),
                                  self.ui_state, "prior.weight_dtype")

            row += 1

        if has_text_encoder:
            # text encoder weight dtype
            components.label(self.scroll_frame, row, 3, "覆盖文本编码器数据类型",
                             tooltip="覆盖文本编码器权重数据类型")
            components.options_kv(self.scroll_frame, row, 4,  self.__create_dtype_options(),
                                  self.ui_state, "text_encoder.weight_dtype")

            row += 1

        if has_text_encoder_1:
            # text encoder 1 weight dtype
            components.label(self.scroll_frame, row, 3, "覆盖文本编码器 1 数据类型",
                             tooltip="覆盖文本编码器 1 权重数据类型")
            components.options_kv(self.scroll_frame, row, 4,  self.__create_dtype_options(),
                                  self.ui_state, "text_encoder.weight_dtype")

            row += 1

        if has_text_encoder_2:
            # text encoder 2 weight dtype
            components.label(self.scroll_frame, row, 3, "覆盖文本编码器 2 数据类型",
                             tooltip="覆盖文本编码器 2 权重数据类型")
            components.options_kv(self.scroll_frame, row, 4,  self.__create_dtype_options(),
                                  self.ui_state, "text_encoder_2.weight_dtype")

            row += 1

        if has_text_encoder_3:
            # text encoder 3 weight dtype
            components.label(self.scroll_frame, row, 3, "覆盖文本编码器 2 数据类型",
                             tooltip="覆盖文本编码器 3 权重数据类型")
            components.options_kv(self.scroll_frame, row, 4,  self.__create_dtype_options(),
                                  self.ui_state, "text_encoder_3.weight_dtype")

            row += 1

        if has_vae:
            # base model
            components.label(self.scroll_frame, row, 0, "VAE 覆盖",
                             tooltip="diffusers 格式的 VAE 模型的目录或 Hugging Face 仓库。可用于覆盖基础模型中包含的 VAE。使用 safetensor VAE 文件会导致错误，即模型无法加载。")
            components.file_entry(
                self.scroll_frame, row, 1, self.ui_state, "vae.model_name",
                path_modifier=lambda x: Path(x).parent.absolute() if x.endswith(".json") else x
            )

            # vae weight dtype
            components.label(self.scroll_frame, row, 3, "覆盖 VAE 数据类型",
                             tooltip="覆盖 vae 权重数据类型")
            components.options_kv(self.scroll_frame, row, 4,  self.__create_dtype_options(),
                                  self.ui_state, "vae.weight_dtype")

            row += 1

        return row

    def __create_effnet_encoder_components(self, row: int):
        # effnet encoder model
        components.label(self.scroll_frame, row, 0, "Effnet 编码器模型",
                         tooltip="Effnet 编码器模型的文件名、目录或 Hugging Face 仓库")
        components.file_entry(
            self.scroll_frame, row, 1, self.ui_state, "effnet_encoder.model_name",
            path_modifier=lambda x: Path(x).parent.absolute() if x.endswith(".json") else x
        )

        # effnet encoder weight dtype
        components.label(self.scroll_frame, row, 3, "覆盖 Effnet 编码器数据类型",
                         tooltip="覆盖 Effnet 编码器权重数据类型")
        components.options_kv(self.scroll_frame, row, 4, self.__create_dtype_options(),
                              self.ui_state, "effnet_encoder.weight_dtype")

        row += 1

        return row

    def __create_decoder_components(
            self,
            row: int,
            has_text_encoder: bool,
    ) -> int:
        # decoder model
        components.label(self.scroll_frame, row, 0, "解码器模型",
                         tooltip="解码器模型的文件名、目录或 Hugging Face 仓库")
        components.file_entry(
            self.scroll_frame, row, 1, self.ui_state, "decoder.model_name",
            path_modifier=lambda x: Path(x).parent.absolute() if x.endswith(".json") else x
        )

        # decoder weight dtype
        components.label(self.scroll_frame, row, 3, "覆盖解码器数据类型",
                         tooltip="覆盖解码器权重数据类型")
        components.options_kv(self.scroll_frame, row, 4, self.__create_dtype_options(),
                              self.ui_state, "decoder.weight_dtype")

        row += 1

        if has_text_encoder:
            # decoder text encoder weight dtype
            components.label(self.scroll_frame, row, 3, "覆盖解码器文本编码器数据类型",
                             tooltip="覆盖解码器文本编码器权重数据类型")
            components.options_kv(self.scroll_frame, row, 4,  self.__create_dtype_options(),
                                  self.ui_state, "decoder_text_encoder.weight_dtype")

            row += 1

        # decoder vqgan weight dtype
        components.label(self.scroll_frame, row, 3, "覆盖解码器 VQGAN 数据类型",
                         tooltip="覆盖解码器 VQGAN 权重数据类型")
        components.options_kv(self.scroll_frame, row, 4, self.__create_dtype_options(),
                              self.ui_state, "decoder_vqgan.weight_dtype")

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
        components.label(self.scroll_frame, row, 0, "模型输出目标",
                         tooltip="保存输出模型的文件名或目录")
        components.file_entry(self.scroll_frame, row, 1, self.ui_state, "output_model_destination", is_output=True)

        # output data type
        components.label(self.scroll_frame, row, 3, "输出数据类型",
                         tooltip="保存输出模型时使用的精度")
        components.options_kv(self.scroll_frame, row, 4, [
            ("float16", DataType.FLOAT_16),
            ("float32", DataType.FLOAT_32),
            ("bfloat16", DataType.BFLOAT_16),
            ("float8", DataType.FLOAT_8),
            ("nfloat4", DataType.NFLOAT_4),
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

        components.label(self.scroll_frame, row, 0, "输出格式",
                         tooltip="保存输出模型时使用的格式")
        components.options_kv(self.scroll_frame, row, 1, formats, self.ui_state, "output_model_format")

        # include config
        components.label(self.scroll_frame, row, 3, "包含配置",
                         tooltip="在最终模型中包含训练配置。仅支持 safetensors 文件。\n"
                                 "None: 不包含任何配置。\n"
                                 "Settings: 包含所有训练设置。\n"
                                 "All: 包含所有设置，包括样本和概念。")
        components.options_kv(self.scroll_frame, row, 4, [
            ("None", ConfigPart.NONE),
            ("Settings", ConfigPart.SETTINGS),
            ("All", ConfigPart.ALL),
        ], self.ui_state, "include_train_config")

        row += 1

        return row
