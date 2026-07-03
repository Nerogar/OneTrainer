import contextlib
import traceback
from uuid import uuid4

from modules.util import create
from modules.util.args.ConvertModelArgs import ConvertModelArgs
from modules.util.config.TrainConfig import QuantizationConfig
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.ModelNames import EmbeddingName, ModelNames
from modules.util.torch_util import torch_gc

import huggingface_hub


class ConvertModelUIController:
    def __init__(
            self,
            model_type: ModelType | None = None,
            base_model_name: str | None = None,
            huggingface_token: str | None = None,
    ):
        self.convert_model_args = ConvertModelArgs.default_values()
        # prefill from the main training window when the tool is opened from there, so the user does not
        # have to re-specify a model type/base model they already picked for training
        if model_type is not None:
            self.convert_model_args.model_type = model_type
        if base_model_name is not None:
            self.convert_model_args.base_model_name = base_model_name
        if huggingface_token is not None:
            self.convert_model_args.huggingface_token = huggingface_token
        self.view = None

    def create_window(self, parent, view_cls):
        self.view = view_cls(parent, self)
        return self.view

    def get_output_formats(self) -> list[tuple[str, ModelFormat]]:
        labels = {
            ModelFormat.SAFETENSORS: "Safetensors",
            ModelFormat.DIFFUSERS_LORA: "Diffusers",
            ModelFormat.KOHYA_LORA: "Kohya",
            ModelFormat.ORIGINAL_LORA: "Original",
            ModelFormat.COMFY_LORA: "Comfy",
            ModelFormat.LEGACY_LORA: "Legacy",
            ModelFormat.DIFFUSERS: "Diffusers",
            ModelFormat.ORIGINAL_SINGLE_FILE: "Original (single file)",
            ModelFormat.ORIGINAL_TRANSFORMER: "Original (transformer only)",
            ModelFormat.COMFY_TRANSFORMER: "Comfy (transformer only)",
            ModelFormat.LEGACY_SAFETENSORS: "Legacy",
        }
        formats = self.convert_model_args.model_type.supported_output_formats(self.convert_model_args.training_method)
        return [(labels[fmt], fmt) for fmt in formats]

    def convert_model(self):
        try:
            self.view.set_converting(True)
            model_loader = create.create_model_loader(
                model_type=self.convert_model_args.model_type,
                training_method=self.convert_model_args.training_method
            )
            model_saver = create.create_model_saver(
                model_type=self.convert_model_args.model_type,
                training_method=self.convert_model_args.training_method
            )

            if self.convert_model_args.huggingface_token != "":
                with contextlib.suppress(ConnectionError):
                    huggingface_hub.login(token=self.convert_model_args.huggingface_token)

            print("Loading model " + self.convert_model_args.input_name)
            if self.convert_model_args.training_method in [TrainingMethod.FINE_TUNE]:
                model = model_loader.load(
                    model_type=self.convert_model_args.model_type,
                    model_names=ModelNames(
                        base_model=self.convert_model_args.input_name,
                    ),
                    weight_dtypes=self.convert_model_args.weight_dtypes(),
                    quantization=QuantizationConfig.default_values(),
                )
            elif self.convert_model_args.training_method in [TrainingMethod.LORA, TrainingMethod.EMBEDDING]:
                model = model_loader.load(
                    model_type=self.convert_model_args.model_type,
                    model_names=ModelNames(
                        base_model=self.convert_model_args.base_model_name or None,
                        lora=self.convert_model_args.input_name,
                        embedding=EmbeddingName(str(uuid4()), self.convert_model_args.input_name),
                    ),
                    weight_dtypes=self.convert_model_args.weight_dtypes(),
                    quantization=QuantizationConfig.default_values(),
                )
            else:
                raise Exception("could not load model: " + self.convert_model_args.input_name)

            print("Saving model " + self.convert_model_args.output_model_destination)
            model_saver.save(
                model=model,
                model_type=self.convert_model_args.model_type,
                output_model_format=self.convert_model_args.output_model_format,
                output_model_destination=self.convert_model_args.output_model_destination,
                dtype=self.convert_model_args.output_dtype.torch_dtype(),
            )
            print("Model converted")
        except Exception:
            traceback.print_exc()

        torch_gc()
        self.view.set_converting(False)
