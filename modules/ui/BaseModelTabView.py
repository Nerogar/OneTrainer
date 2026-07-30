
from abc import ABC, abstractmethod

from modules.util import path_util
from modules.util.enum.ConfigPart import ConfigPart
from modules.util.enum.DataType import DataType
from modules.util.enum.PathIOType import PathIOType

from huggingface_hub.constants import HF_HUB_CACHE


class BaseModelTabView(ABC):
    def __init__(self, components):
        self.components = components

    @abstractmethod
    def _make_svd_frames(self, parent, row: int):
        # Create and place SVDQuant label+entry subframes; return (label_frame, entry_frame).
        pass

    def build_content(self, frame, controller, ui_state):
        model_type = controller.train_config.model_type
        parts = model_type.model_parts()

        row = 0
        row = self.__create_base_dtype_components(frame, row, ui_state)
        row = self.__create_base_components(
            frame,
            row,
            controller,
            ui_state,
            has_unet="unet" in parts,
            has_prior="prior" in parts,
            allow_override_prior=model_type.is_stable_cascade(),
            has_transformer="transformer" in parts,
            allow_override_transformer=controller.supports_override_transformer(),
            has_unconditional_transformer="unconditional_transformer" in parts,
            has_text_encoder=not model_type.has_multiple_text_encoders(),
            has_text_encoder_1=model_type.has_multiple_text_encoders(),
            has_text_encoder_2="text_encoder_2" in parts,
            has_text_encoder_3="text_encoder_3" in parts,
            has_text_encoder_4="text_encoder_4" in parts,
            allow_override_text_encoder_4="text_encoder_4" in parts,
            has_vae="vae" in parts,
        )
        if "effnet_encoder" in parts:
            row = self.__create_effnet_encoder_components(frame, row, ui_state)
        if "decoder" in parts:
            row = self.__create_decoder_components(frame, row, ui_state, "decoder_text_encoder" in parts)

        self.__create_output_components(
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
        self.components.label(frame, row, 0, "Hugging Face令牌",
                         tooltip="Enter your Hugging Face access token if you have used a protected Hugging Face repository below.\nThis value is stored separately, not saved to your configuration file. "
                                 "Go to https://huggingface.co/settings/tokens to create an access token.",
                         wide_tooltip=True)
        self.components.entry(frame, row, 1, ui_state, "secrets.huggingface_token")

        # offline mode
        self.components.label(frame, row, 3, "离线模式",
                         tooltip="Skip the Hugging Face login and resolve every model from the local cache only. "
                                 "Enable this when you have no internet connection; only already-downloaded models can be loaded.",
                         wide_tooltip=True)
        self.components.switch(frame, row, 4, ui_state, "offline_mode")

        row += 1

        # huggingface cache directory
        self.components.label(frame, row, 0, "Hugging Face缓存目录",
                         tooltip="Directory used to cache Hugging Face model downloads. "
                                 "Leave empty to use the default Hugging Face cache directory shown as the placeholder.",
                         wide_tooltip=True)
        self.components.path_entry(
            frame, row, 1, ui_state, "huggingface_cache_dir",
            mode="dir", placeholder=HF_HUB_CACHE,
        )

        row += 1

        # base model
        self.components.label(frame, row, 0, "基础模型",
                         tooltip="基础模型文件名、目录或Hugging Face仓库")
        self.components.path_entry(
            frame, row, 1, ui_state, "base_model_name",
            mode="file", path_modifier=path_util.json_path_modifier
        )

        # compile
        self.components.label(frame, row, 3, "编译Transformer块",
                         tooltip="使用torch.compile和Triton加速训练，如有兼容问题请禁用")
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
            has_unconditional_transformer: bool = False,
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
            self.components.label(frame, row, 3, "UNet数据类型",
                             tooltip="UNet权重数据类型")
            self.components.options_kv(frame, row, 4, self.__create_dtype_options(include_a8=True),
                                  ui_state, "unet.weight_dtype")

            row += 1

        if has_prior:
            if allow_override_prior:
                # prior model
                self.components.label(frame, row, 0, "Prior模型",
                                 tooltip="Prior模型路径")
                self.components.path_entry(
                    frame, row, 1, ui_state, "prior.model_name",
                    mode="file", path_modifier=path_util.json_path_modifier
                )

            # prior weight dtype
            self.components.label(frame, row, 3, "Prior数据类型",
                             tooltip="Prior权重数据类型")
            self.components.options_kv(frame, row, 4, self.__create_dtype_options(),
                                  ui_state, "prior.weight_dtype")

            row += 1

        if has_transformer:
            if allow_override_transformer:
                # transformer model
                self.components.label(frame, row, 0, "Override Transformer / GGUF",
                                 tooltip="覆盖基础模型的Transformer，支持safetensors和GGUF")
                self.components.path_entry(
                    frame, row, 1, ui_state, "transformer.model_name",
                    mode="file", path_modifier=path_util.json_path_modifier
                )

            # transformer weight dtype
            self.components.label(frame, row, 3, "Transformer数据类型",
                             tooltip="Transformer权重数据类型")
            self.components.options_kv(frame, row, 4, self.__create_dtype_options(include_gguf=True, include_a8=True),
                                  ui_state, "transformer.weight_dtype")

            row += 1

        if has_unconditional_transformer:
            # unconditional transformer weight dtype
            self.components.label(frame, row, 3, "无条件Transformer数据类型",
                             tooltip="无条件Transformer权重数据类型，用于CFG负分支")
            self.components.options_kv(frame, row, 4, self.__create_dtype_options(include_a8=True),
                                  ui_state, "unconditional_transformer.weight_dtype")

            row += 1

        presets = controller.get_presets()

        self.components.label(frame, row, 0, "量化")
        self.components.layer_filter_entry(frame, row, 1, ui_state,
            preset_var_name="quantization.layer_filter_preset", presets=presets,
            preset_label="量化层过滤器",
            preset_tooltip="选择量化层预设，量化某些层可能降低模型质量",
            entry_var_name="quantization.layer_filter",
            entry_tooltip="逗号分隔的量化层列表，支持正则表达式",
            regex_var_name="quantization.layer_filter_regex",
            regex_tooltip="启用后层过滤器使用正则匹配，否则使用子串匹配",
            frame_color="transparent",
        )

        # SVDQuant - create vertical grids to match the size of layer_filter_entry
        svd_label_frame, svd_entry_frame = self._make_svd_frames(frame, row)
        self.components.label(svd_label_frame, 0, 0, "SVDQuant",
                         tooltip="SVDQuant权重分解的数据类型")
        self.components.options_kv(svd_entry_frame, 0, 0, [("disabled", DataType.NONE), ("float32", DataType.FLOAT_32), ("bfloat16", DataType.BFLOAT_16)],
                              ui_state, "quantization.svd_dtype")
        self.components.label(svd_label_frame, 1, 0, "SVDQuant秩",
                         tooltip="SVDQuant权重分解的秩")
        self.components.entry(svd_entry_frame, 1, 0, ui_state, "quantization.svd_rank")
        row += 1

        if has_text_encoder:
            # text encoder weight dtype
            self.components.label(frame, row, 3, "文本编码器数据类型",
                             tooltip="文本编码器权重数据类型")
            self.components.options_kv(frame, row, 4, self.__create_dtype_options(),
                                  ui_state, "text_encoder.weight_dtype")

            row += 1

        if has_text_encoder_1:
            # text encoder 1 weight dtype
            self.components.label(frame, row, 3, "Text Encoder 1 Data Type",
                             tooltip="文本编码器1权重数据类型")
            self.components.options_kv(frame, row, 4, self.__create_dtype_options(),
                                  ui_state, "text_encoder.weight_dtype")

            row += 1

        if has_text_encoder_2:
            # text encoder 2 weight dtype
            self.components.label(frame, row, 3, "Text Encoder 2 Data Type",
                             tooltip="文本编码器2权重数据类型")
            self.components.options_kv(frame, row, 4, self.__create_dtype_options(),
                                  ui_state, "text_encoder_2.weight_dtype")

            row += 1

        if has_text_encoder_3:
            # text encoder 3 weight dtype
            self.components.label(frame, row, 3, "Text Encoder 3 Data Type",
                             tooltip="文本编码器3权重数据类型")
            self.components.options_kv(frame, row, 4, self.__create_dtype_options(),
                                  ui_state, "text_encoder_3.weight_dtype")

            row += 1

        if has_text_encoder_4:
            if allow_override_text_encoder_4:
                # text encoder 4 weight dtype
                self.components.label(frame, row, 0, "Text Encoder 4 Override",
                                 tooltip="文本编码器4模型路径")
                self.components.path_entry(
                    frame, row, 1, ui_state, "text_encoder_4.model_name",
                    mode="file", path_modifier=path_util.json_path_modifier
                )

            # text encoder 4 weight dtype
            self.components.label(frame, row, 3, "Text Encoder 4 Data Type",
                             tooltip="文本编码器4权重数据类型")
            self.components.options_kv(frame, row, 4, self.__create_dtype_options(),
                                  ui_state, "text_encoder_4.weight_dtype")

            row += 1

        if has_vae:
            # base model
            self.components.label(frame, row, 0, "VAE覆盖",
                             tooltip="diffusers格式的VAE模型目录或Hugging Face仓库，用于覆盖基础模型的VAE")
            self.components.path_entry(
                frame, row, 1, ui_state, "vae.model_name",
                mode="file", path_modifier=path_util.json_path_modifier
            )

            # vae weight dtype
            self.components.label(frame, row, 3, "VAE数据类型",
                             tooltip="VAE权重数据类型")
            self.components.options_kv(frame, row, 4, self.__create_dtype_options(),
                                  ui_state, "vae.weight_dtype")

            row += 1

        return row

    def __create_effnet_encoder_components(self, frame, row: int, ui_state) -> int:
        # effnet encoder model
        self.components.label(frame, row, 0, "Effnet编码器模型",
                         tooltip="Effnet编码器模型路径")
        self.components.path_entry(
            frame, row, 1, ui_state, "effnet_encoder.model_name",
            mode="file", path_modifier=path_util.json_path_modifier
        )

        # effnet encoder weight dtype
        self.components.label(frame, row, 3, "Effnet编码器数据类型",
                         tooltip="Effnet编码器权重数据类型")
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
        self.components.label(frame, row, 0, "解码器模型",
                         tooltip="解码器模型路径")
        self.components.path_entry(
            frame, row, 1, ui_state, "decoder.model_name",
            mode="file", path_modifier=path_util.json_path_modifier
        )

        # decoder weight dtype
        self.components.label(frame, row, 3, "解码器数据类型",
                         tooltip="解码器权重数据类型")
        self.components.options_kv(frame, row, 4, self.__create_dtype_options(),
                              ui_state, "decoder.weight_dtype")

        row += 1

        if has_text_encoder:
            # decoder text encoder weight dtype
            self.components.label(frame, row, 3, "解码器文本编码器数据类型",
                             tooltip="解码器文本编码器权重数据类型")
            self.components.options_kv(frame, row, 4, self.__create_dtype_options(),
                                  ui_state, "decoder_text_encoder.weight_dtype")

            row += 1

        # decoder vqgan weight dtype
        self.components.label(frame, row, 3, "解码器VQGAN数据类型",
                         tooltip="解码器VQGAN权重数据类型")
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
        self.components.label(frame, row, 0, "模型输出目标",
                         tooltip="输出模型保存的文件名或目录")
        self.components.path_entry(
            frame, row, 1, ui_state, "output_model_destination",
            mode="file",
            io_type=PathIOType.MODEL,
        )

        # output data type
        self.components.label(frame, row, 3, "输出数据类型",
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

        self.components.label(frame, row, 0, "输出格式",
                         tooltip="保存输出模型的格式")
        self.components.options_kv(frame, row, 1, formats, ui_state, "output_model_format")

        # include config
        self.components.label(frame, row, 3, "包含配置",
                         tooltip="Include the training configuration in the final model. Only supported for safetensors files. "
                                 "None: No config is included. "
                                 "Settings: All training settings are included. "
                                 "全部：包含所有设置、采样和数据集")
        self.components.options_kv(frame, row, 4, [
            ("无", ConfigPart.NONE),
            ("设置", ConfigPart.SETTINGS),
            ("All", ConfigPart.ALL),
        ], ui_state, "include_train_config")

        row += 1

        return row
