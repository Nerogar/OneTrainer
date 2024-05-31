import traceback
from pathlib import Path
from uuid import uuid4

import customtkinter as ctk
import torch

from modules.util import create
from modules.util.ModelNames import ModelNames, EmbeddingName
from modules.util.args.ConvertModelArgs import ConvertModelArgs
from modules.util.enum.DataType import DataType
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.ui import components
from modules.util.ui.UIState import UIState
from modules.util.torch_util import torch_gc


class ConvertModelUI(ctk.CTkToplevel):
    def __init__(self, parent, *args, **kwargs):
        ctk.CTkToplevel.__init__(self, parent, *args, **kwargs)
        self.parent = parent

        self.title("Convert models")
        self.geometry("550x350")
        self.resizable(False, False)
        self.wait_visibility()
        self.focus_set()

        self.convert_model_args = ConvertModelArgs.default_values()
        self.ui_state = UIState(self, self.convert_model_args)

        self.frame = ctk.CTkFrame(self, width=600, height=300)
        self.frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        self.frame.grid_columnconfigure(0, weight=0)
        self.frame.grid_columnconfigure(1, weight=1)

        self.button = None
        self.main_frame(self.frame)

        self.frame.pack(fill="both", expand=True)

    def main_frame(self, master):
        # model type
        components.label(master, 0, 0, "Model Type",
                         tooltip="Type of the model")
        components.options_kv(master, 0, 1, [
            ("Stable Diffusion 1.5", ModelType.STABLE_DIFFUSION_15),
            ("Stable Diffusion 1.5 Inpainting", ModelType.STABLE_DIFFUSION_15_INPAINTING),
            ("Stable Diffusion 2.0", ModelType.STABLE_DIFFUSION_20),
            ("Stable Diffusion 2.0 Inpainting", ModelType.STABLE_DIFFUSION_20_INPAINTING),
            ("Stable Diffusion 2.1", ModelType.STABLE_DIFFUSION_21),
            ("Stable Diffusion XL 1.0 Base", ModelType.STABLE_DIFFUSION_XL_10_BASE),
            ("Stable Diffusion XL 1.0 Base Inpainting", ModelType.STABLE_DIFFUSION_XL_10_BASE_INPAINTING),
            ("Wuerstchen v2", ModelType.WUERSTCHEN_2),
            ("Stable Cascade", ModelType.STABLE_CASCADE_1),
            ("PixArt Alpha", ModelType.PIXART_ALPHA),
            ("PixArt Sigma", ModelType.PIXART_SIGMA),
        ], self.ui_state, "model_type")

        # training method
        components.label(master, 1, 0, "Model Type",
                         tooltip="The type of model to convert")
        components.options_kv(master, 1, 1, [
            ("Base Model", TrainingMethod.FINE_TUNE),
            ("LoRA", TrainingMethod.LORA),
            ("Embedding", TrainingMethod.EMBEDDING),
        ], self.ui_state, "training_method")

        # input name
        components.label(master, 2, 0, "Input name",
                         tooltip="Filename, directory or hugging face repository of the base model")
        components.file_entry(
            master, 2, 1, self.ui_state, "input_name",
            path_modifier=lambda x: Path(x).parent.absolute() if x.endswith(".json") else x
        )

        # output data type
        components.label(master, 3, 0, "Output Data Type",
                         tooltip="Precision to use when saving the output model")
        components.options_kv(master, 3, 1, [
            ("float32", DataType.FLOAT_32),
            ("float16", DataType.FLOAT_16),
            ("bfloat16", DataType.BFLOAT_16),
        ], self.ui_state, "output_dtype")

        # output format
        components.label(master, 4, 0, "Output Format",
                         tooltip="Format to use when saving the output model")
        components.options_kv(master, 4, 1, [
            ("Safetensors", ModelFormat.SAFETENSORS),
            ("Diffusers", ModelFormat.DIFFUSERS),
            ("Checkpoint", ModelFormat.CKPT),
        ], self.ui_state, "output_model_format")

        # output model destination
        components.label(master, 5, 0, "Model Output Destination",
                         tooltip="Filename or directory where the output model is saved")
        components.file_entry(master, 5, 1, self.ui_state, "output_model_destination", is_output=True)

        self.button = components.button(master, 6, 1, "Convert", self.convert_model)

    def convert_model(self):
        try:
            self.button.configure(state="disabled")
            model_loader = create.create_model_loader(
                model_type=self.convert_model_args.model_type,
                training_method=self.convert_model_args.training_method
            )
            model_saver = create.create_model_saver(
                model_type=self.convert_model_args.model_type,
                training_method=self.convert_model_args.training_method
            )

            print("Loading model " + self.convert_model_args.input_name)
            if self.convert_model_args.training_method in [TrainingMethod.FINE_TUNE]:
                model = model_loader.load(
                    model_type=self.convert_model_args.model_type,
                    model_names=ModelNames(
                        base_model=self.convert_model_args.input_name,
                    ),
                    weight_dtypes=self.convert_model_args.weight_dtypes(),
                )
            elif self.convert_model_args.training_method in [TrainingMethod.LORA, TrainingMethod.EMBEDDING]:
                model = model_loader.load(
                    model_type=self.convert_model_args.model_type,
                    model_names=ModelNames(
                        lora=self.convert_model_args.input_name,
                        embedding=EmbeddingName(str(uuid4()), self.convert_model_args.input_name),
                    ),
                    weight_dtypes=self.convert_model_args.weight_dtypes(),
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
        except:
            traceback.print_exc()

        torch_gc()
        self.button.configure(state="normal")
