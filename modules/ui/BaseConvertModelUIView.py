from modules.util import path_util
from modules.util.enum.DataType import DataType
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.ModelType import ModelType
from modules.util.enum.PathIOType import PathIOType
from modules.util.enum.TrainingMethod import TrainingMethod


class BaseConvertModelUIView:
    def __init__(self, components):
        self.components = components

    def build_content(self, frame, controller, ui_state):
        # model type
        self.components.label(frame, 0, 0, "Model Type",
                         tooltip="Type of the model")
        self.components.options_kv(frame, 0, 1, [ #TODO simplify
            ("Stable Diffusion 1.5", ModelType.STABLE_DIFFUSION_15),
            ("Stable Diffusion 1.5 Inpainting", ModelType.STABLE_DIFFUSION_15_INPAINTING),
            ("Stable Diffusion 2.0", ModelType.STABLE_DIFFUSION_20),
            ("Stable Diffusion 2.0 Inpainting", ModelType.STABLE_DIFFUSION_20_INPAINTING),
            ("Stable Diffusion 2.1", ModelType.STABLE_DIFFUSION_21),
            ("Stable Diffusion 3", ModelType.STABLE_DIFFUSION_3),
            ("Stable Diffusion 3.5", ModelType.STABLE_DIFFUSION_35),
            ("Stable Diffusion XL 1.0 Base", ModelType.STABLE_DIFFUSION_XL_10_BASE),
            ("Stable Diffusion XL 1.0 Base Inpainting", ModelType.STABLE_DIFFUSION_XL_10_BASE_INPAINTING),
            ("Wuerstchen v2", ModelType.WUERSTCHEN_2),
            ("Stable Cascade", ModelType.STABLE_CASCADE_1),
            ("PixArt Alpha", ModelType.PIXART_ALPHA),
            ("PixArt Sigma", ModelType.PIXART_SIGMA),
            ("Flux Dev", ModelType.FLUX_DEV_1),
            ("Flux Fill Dev", ModelType.FLUX_FILL_DEV_1),
            ("Flux 2", ModelType.FLUX_2),
            ("Hunyuan Video", ModelType.HUNYUAN_VIDEO),
            ("Chroma1", ModelType.CHROMA_1), #TODO does this just work? HiDream is not here
            ("QwenImage", ModelType.QWEN), #TODO does this just work? HiDream is not here
            ("Krea 2", ModelType.KREA_2),
            ("ZImage", ModelType.Z_IMAGE),
        ], ui_state, "model_type")

        # training method
        self.components.label(frame, 1, 0, "Model Type",
                         tooltip="The type of model to convert")
        self.components.options_kv(frame, 1, 1, [
            ("Base Model", TrainingMethod.FINE_TUNE),
            ("LoRA", TrainingMethod.LORA),
            ("Embedding", TrainingMethod.EMBEDDING),
        ], ui_state, "training_method")

        # input name
        self.components.label(frame, 2, 0, "Input name",
                         tooltip="Filename, directory or hugging face repository of the base model")
        self.components.path_entry(
            frame, 2, 1, ui_state, "input_name",
            mode="file", path_modifier=path_util.json_path_modifier
        )

        # output data type
        self.components.label(frame, 3, 0, "Output Data Type",
                         tooltip="Precision to use when saving the output model")
        self.components.options_kv(frame, 3, 1, [
            ("float32", DataType.FLOAT_32),
            ("float16", DataType.FLOAT_16),
            ("bfloat16", DataType.BFLOAT_16),
        ], ui_state, "output_dtype")

        # output format
        self.components.label(frame, 4, 0, "Output Format",
                         tooltip="Format to use when saving the output model")
        self.components.options_kv(frame, 4, 1, [
            ("Safetensors", ModelFormat.SAFETENSORS),
            ("Diffusers", ModelFormat.DIFFUSERS),
        ], ui_state, "output_model_format")

        # output model destination
        self.components.label(frame, 5, 0, "Model Output Destination",
                         tooltip="Filename or directory where the output model is saved")
        self.components.path_entry(
            frame, 5, 1, ui_state, "output_model_destination",
            mode="file",
            io_type=PathIOType.MODEL,
        )

        self.button = self.components.button(frame, 6, 1, "Convert", controller.convert_model)
