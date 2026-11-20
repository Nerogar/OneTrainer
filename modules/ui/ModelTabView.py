from modules.ui.ModelTabController import ModelTabController
from modules.util import path_util
from modules.util.enum.ConfigPart import ConfigPart
from modules.util.enum.DataType import DataType
from modules.util.enum.PathIOType import PathIOType
from modules.util.enum.RunNameMode import RunNameMode
from modules.util.ui import pyside6_components
from modules.util.ui.pyside6_util import fit_form

from huggingface_hub.constants import HF_HUB_CACHE
from PySide6.QtWidgets import QWidget


class BaseModelTabView:
    def __init__(self, components):
        self.components = components

    def _create_dtype_options(self, include_gguf: bool = False, include_a8: bool = False,
                              include_compressed: bool = False) -> list[tuple[str, DataType]]:
        options = [
            ("float32", DataType.FLOAT_32),
            ("bfloat16", DataType.BFLOAT_16),
            ("float16", DataType.FLOAT_16),
            ("float8 (W8)", DataType.FLOAT_8),
            ("nfloat4", DataType.NFLOAT_4),
        ]

        if include_a8:
            options.append(("float W8A8", DataType.FLOAT_W8A8))
            if include_compressed:
                options.append(("float W8A8 compressed", DataType.FLOAT_W8A8_COMPRESSED))
            options.append(("int W8A8", DataType.INT_W8A8))
            if include_compressed:
                options.append(("int W8A8 compressed", DataType.INT_W8A8_COMPRESSED))

            options += [
                ("bfloat16 A8 int", DataType.BFLOAT_16_A8_INT),
                ("bfloat16 A8 float", DataType.BFLOAT_16_A8_FLOAT),
            ]

        if include_gguf:
            options.append(("GGUF", DataType.GGUF))
            if include_a8:
                options += [
                    ("GGUF A8 int", DataType.GGUF_A8_INT),
                    ("GGUF A8 float", DataType.GGUF_A8_FLOAT),
                ]

        return options

    def _build_model_source(self, frame, controller, ui_state):
        model_type = controller.train_config.model_type
        parts = model_type.model_parts()

        self.components.label(frame, 0, 0, "HF Token",
                              tooltip="Enter your Hugging Face access token if you have used a protected Hugging Face repository below.\nThis value is stored separately, not saved to your configuration file. "
                                      "Go to https://huggingface.co/settings/tokens to create an access token.",
                              wide_tooltip=True)
        self.components.entry(frame, 0, 1, ui_state, "secrets.huggingface_token")

        self.components.label(frame, 1, 0, "HF Cache Dir",
                              tooltip="Directory used to cache Hugging Face model downloads. "
                                      "Leave empty to use the default Hugging Face cache directory shown as the placeholder.",
                              wide_tooltip=True)
        self.components.path_entry(frame, 1, 1, ui_state, "huggingface_cache_dir",
                                   mode="dir", placeholder=HF_HUB_CACHE)

        self.components.label(frame, 2, 0, "Offline Mode",
                              tooltip="Skip the Hugging Face login and resolve every model from the local cache only. "
                                      "Enable this when you have no internet connection; only already-downloaded models can be loaded.",
                              wide_tooltip=True)
        self.components.switch(frame, 2, 1, ui_state, "offline_mode")

        self.components.label(frame, 3, 0, "Base Model",
                              tooltip="Filename, directory or Hugging Face repository of the base model")
        self.components.path_entry(frame, 3, 1, ui_state, "base_model_name",
                                   mode="file", path_modifier=path_util.json_path_modifier)

        self.components.label(frame, 4, 0, "Stream From Disk",
                              tooltip="Streams tensors from disk in small groups, applying quantization as they load when enabled, to reduce peak RAM usage during loading. Turn off if you encounter compatibility issues.")
        self.components.switch(frame, 4, 1, ui_state, "stream_from_disk")

        row = 5
        if "transformer" in parts and controller.supports_override_transformer():
            self.components.label(frame, row, 0, "Override Transformer / GGUF",
                                  tooltip="Can be used to override the transformer in the base model. Safetensors and GGUF files are supported, local and on Huggingface. If a GGUF file is used, the DataType must also be set to GGUF")
            self.components.path_entry(frame, row, 1, ui_state, "transformer.model_name",
                                       mode="file", path_modifier=path_util.json_path_modifier)
            row += 1

        if "vae" in parts:
            self.components.label(frame, row, 0, "Override VAE",
                                  tooltip="Directory or Hugging Face repository of a VAE model in diffusers format. Can be used to override the VAE included in the base model. Using a safetensor VAE file will cause an error that the model cannot be loaded.")
            self.components.path_entry(frame, row, 1, ui_state, "vae.model_name",
                                       mode="file", path_modifier=path_util.json_path_modifier)
            row += 1

        for part, label, tooltip, available in (
            ("prior", "Prior Model", "Filename, directory or Hugging Face repository of the prior model",
             "prior" in parts and model_type.is_stable_cascade()),
            ("text_encoder_4", "Text Encoder 4 Override", "Filename, directory or Hugging Face repository of the text encoder 4 model",
             "text_encoder_4" in parts),
            ("effnet_encoder", "Effnet Encoder Model", "Filename, directory or Hugging Face repository of the effnet encoder model",
             "effnet_encoder" in parts),
            ("decoder", "Decoder Model", "Filename, directory or Hugging Face repository of the decoder model",
             "decoder" in parts),
        ):
            if available:
                self.components.label(frame, row, 0, label, tooltip=tooltip)
                self.components.path_entry(frame, row, 1, ui_state, f"{part}.model_name",
                                           mode="file", path_modifier=path_util.json_path_modifier)
                row += 1

    def _build_output(self, frame, controller, ui_state):
        self.components.label(frame, 0, 0, "Run Name",
                              tooltip="Filename for the final model and prefix for saves, backups, samples and Tensorboard. "
                                      "The output format determines the extension.")
        name_frame = self.components.inline_frame(frame, 0, 1)
        self.components.options_kv(name_frame, 0, 0, [
            ("Default", RunNameMode.DEFAULT), ("Friendly", RunNameMode.FRIENDLY), ("Custom", RunNameMode.CUSTOM),
        ], ui_state, "run_name_mode")
        self.components.entry(name_frame, 0, 1, ui_state, "run_name")

        self.components.label(frame, 1, 0, "Final Output Dir",
                              tooltip="Directory where the final trained model is saved")
        self.components.path_entry(frame, 1, 1, ui_state, "final_output_dir",
                                   mode="dir", io_type=PathIOType.OUTPUT, required=True)

        self.components.label(frame, 2, 0, "Output Format",
                              tooltip="Format to use when saving the output model")
        self.components.options_kv(frame, 2, 1, controller.get_output_formats(), ui_state, "output_model_format")

        self.components.label(frame, 3, 0, "Include Config",
                              tooltip="Include the training configuration in the final model. Only supported for safetensors files. "
                                      "None: No config is included. "
                                      "Settings: All training settings are included. "
                                      "All: All settings, including the samples and concepts are included.")
        self.components.options_kv(frame, 3, 1, [
            ("None", ConfigPart.NONE),
            ("Settings", ConfigPart.SETTINGS),
            ("All", ConfigPart.ALL),
        ], ui_state, "include_train_config")

    def _build_precision(self, frame, controller, ui_state):
        model_type = controller.train_config.model_type
        parts = model_type.model_parts()
        include_compressed = model_type.supports_compression()

        row = 0
        self.components.layer_filter_entry(frame, row, 0, ui_state,
            preset_var_name="quantization.layer_filter_preset", presets=controller.get_presets(),
            preset_label="Quantization Layer Filter",
            preset_tooltip="Select a preset defining which layers to quantize. Quantization of certain layers can decrease model quality. Only applies to the transformer/unet",
            entry_var_name="quantization.layer_filter",
            entry_tooltip="Comma-separated list of layers to quantize. Regular expressions (if toggled) are supported. Any model layer with a matching name will be quantized",
            regex_var_name="quantization.layer_filter_regex",
            regex_tooltip="If enabled, layer filter patterns are interpreted as regular expressions. Otherwise, simple substring matching is used.",
            colspan=2,
        )
        row += 1

        self.components.label(frame, row, 0, "Compile",
                              tooltip="Uses torch.compile and Triton to significantly speed up training. Only applies to transformer/unet. Disable in case of compatibility issues.")
        self.components.switch(frame, row, 1, ui_state, "compile")
        row += 1

        def part_dtype(part, name, *, a8=False, gguf=False, tooltip=None):
            nonlocal row
            choices = self._create_dtype_options(include_a8=a8, include_gguf=gguf, include_compressed=include_compressed)
            self.components.labeled_field(frame, ui_state, row, self.components.options_kv, f"{part}.weight_dtype",
                                          f"{name} Data Type", tooltip or f"The {name.lower()} weight data type",
                                          values=choices)
            row += 1

        denoiser = model_type.denoising_model_part()
        if denoiser == "prior":
            part_dtype(denoiser, "Prior")
        else:
            transformer = denoiser == "transformer"
            part_dtype(denoiser, "Transformer" if transformer else "UNet", a8=True, gguf=transformer)
            self.components.labeled_field(frame, ui_state, row, self.components.options_kv, "quantization.fallback_dtype",
                                          "Fallback Data Type", "The weight data type used for layers excluded by the "
                                          "quantization layer filter. Can itself be a quantized type.",
                                          values=self._create_dtype_options(include_a8=True, include_compressed=include_compressed))
            row += 1

        self.components.label(frame, row, 0, "SVDQuant",
                              tooltip="What datatype to use for SVDQuant weights decomposition.")
        self.components.options_kv(frame, row, 1, [
            ("disabled", DataType.NONE), ("float32", DataType.FLOAT_32), ("bfloat16", DataType.BFLOAT_16),
        ], ui_state, "quantization.svd_dtype")
        row += 1

        self.components.label(frame, row, 0, "SVDQuant Rank", tooltip="Rank for SVDQuant weights decomposition")
        self.components.entry(frame, row, 1, ui_state, "quantization.svd_rank", max_width=5)
        row += 1

        encoders = model_type.text_encoder_parts()
        for index, part in enumerate(encoders, start=1):
            part_dtype(part, f"Text Encoder {index}" if len(encoders) > 1 else "Text Encoder")
        if "vae" in parts:
            part_dtype("vae", "VAE")
        if "unconditional_transformer" in parts:
            part_dtype("unconditional_transformer", "Unconditional Transformer", a8=True,
                       tooltip="The weight data type of the unconditional transformer, used for the negative branch of CFG during sampling")
        for part, name in (("effnet_encoder", "Effnet Encoder"), ("decoder", "Decoder"),
                           ("decoder_text_encoder", "Decoder Text Encoder"), ("decoder_vqgan", "Decoder VQGAN")):
            if part in parts:
                part_dtype(part, name)

        self.components.label(frame, row, 0, "Output Data Type", tooltip="Precision to use when saving the output model")
        self.components.options_kv(frame, row, 1, [
            ("float16", DataType.FLOAT_16),
            ("float32", DataType.FLOAT_32),
            ("bfloat16", DataType.BFLOAT_16),
            ("float8", DataType.FLOAT_8),
            ("nfloat4", DataType.NFLOAT_4),
        ], ui_state, "output_dtype")


class ModelTabView(BaseModelTabView, QWidget):
    def __init__(self, master, controller: ModelTabController, ui_state):
        QWidget.__init__(self, master)
        BaseModelTabView.__init__(self, pyside6_components)

        self.master = master
        self.controller = controller
        self.ui_state = ui_state
        self.scroll_frame = None
        layout = self.components._layout(self)
        layout.setRowStretch(0, 1)
        layout.setColumnStretch(0, 1)
        self.refresh_ui()

    def refresh_ui(self):
        if self.scroll_frame is not None:
            self.components._layout(self).removeWidget(self.scroll_frame)
            self.scroll_frame.hide()
            self.scroll_frame.deleteLater()

        scroll, frame = self.components.scrollable_frame(self)
        self.components._layout(self).addWidget(scroll, 0, 0)
        self.scroll_frame = scroll

        layout = self.components._layout(frame)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)
        left = self.components.inline_frame(frame, 0, 0)
        self.components._layout(left).setColumnStretch(0, 1)

        source = self.components.form_section(left, 0, 0, "Model Source")
        output = self.components.form_section(left, 1, 0, "Output")
        precision = self.components.form_section(frame, 0, 1, "Precision & Performance")
        self._build_model_source(source, self.controller, self.ui_state)
        self._build_output(output, self.controller, self.ui_state)
        self._build_precision(precision, self.controller, self.ui_state)

        for section in (source, output, precision):
            fit_form(section)
        self.components._pack_form(left)
        self.components._pack_form(frame)
