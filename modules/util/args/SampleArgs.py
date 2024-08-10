import argparse
from typing import Any

from modules.util.ModelNames import ModelNames, EmbeddingName
from modules.util.ModelWeightDtypes import ModelWeightDtypes
from modules.util.args.BaseArgs import BaseArgs
from modules.util.enum.DataType import DataType
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod


class SampleArgs(BaseArgs):
    model_type: ModelType
    weight_dtype: DataType
    base_model_name: str
    embedding_name: str
    prompt: str
    negative_prompt: str
    height: int
    width: int
    destination: str
    text_encoder_layer_skip: int
    sample_inpainting: bool
    base_image_path:str
    mask_image_path:str

    def __init__(self, data: list[(str, Any, type, bool)]):
        super(SampleArgs, self).__init__(data)

    def weight_dtypes(self) -> ModelWeightDtypes:
        return ModelWeightDtypes.from_single_dtype(self.weight_dtype)

    def model_names(self) -> ModelNames:
        return ModelNames(
            base_model=self.base_model_name,
            lora="",
            additional_embeddings=[EmbeddingName("", self.embedding_name)] if self.embedding_name else [],
        )

    @staticmethod
    def parse_args() -> 'SampleArgs':
        parser = argparse.ArgumentParser(description="One Trainer Sampling Script.")

        # @formatter:off

        parser.add_argument("--model-type", type=ModelType, required=True, dest="model_type", help="Type of the base model", choices=list(ModelType))
        parser.add_argument("--weight-dtype", type=DataType, required=False, default=DataType.FLOAT_32, dest="weight_dtype", help="The data type to use for weights during sampling", choices=list(DataType))
        parser.add_argument("--base-model-name", type=str, required=True, dest="base_model_name", help="The base model to sample from")
        parser.add_argument("--embedding-name", type=str, required=False, default="", dest="embedding_name", help="An embedding to use during sampling")
        parser.add_argument("--prompt", type=str, required=True, dest="prompt", help="The prompt for sampling")
        parser.add_argument("--negative-prompt", type=str, required=False, default="", dest="negative_prompt", help="The negative prompt for sampling")
        parser.add_argument("--height", type=int, required=False, default=512, dest="height", help="Height of the image in pixel")
        parser.add_argument("--width", type=int, required=False, default=512, dest="width", help="Width of the image in pixel")
        parser.add_argument("--destination", type=str, required=True, dest="destination", help="The destination to save the output")
        parser.add_argument("--text-encoder-layer-skip", type=int, required=False, default=0, dest="text_encoder_layer_skip", help="Skip last layers of the text encoder")
        parser.add_argument("--sample-inpainting", action="store_true", required=False, default=False, dest="sample_inpainting", help="Enables inpainting sampling. Only available when sampling from an inpainting model.")
        parser.add_argument("--base-image-path", type=str, required=False, default="", dest="base_image_path", help="The base image used when inpainting")
        parser.add_argument("--mask-image-path", type=str, required=False, default="", dest="mask_image_path", help="The mask used when inpainting.")

        # @formatter:on

        args = SampleArgs.default_values()
        args.from_dict(vars(parser.parse_args()))
        return args

    @staticmethod
    def default_values() -> 'SampleArgs':
        data = []

        # name, default value, data type, nullable
        data.append(("model_type", TrainingMethod.FINE_TUNE, TrainingMethod, False))
        data.append(("weight_dtype", DataType.FLOAT_32, DataType, False))
        data.append(("base_model_name", "", str, False))
        data.append(("embedding_name", None, str, True))
        data.append(("prompt", "", str, False))
        data.append(("negative_prompt", "", str, False))
        data.append(("height", 512, int, False))
        data.append(("width", 512, int, False))
        data.append(("destination", "", str, False))
        data.append(("text_encoder_layer_skip", 0, int, False))
        data.append(("sample_inpainting", False, bool, False))
        data.append(("base_image_path", "", str, False))
        data.append(("mask_image_path", "", str, False))

        return SampleArgs(data)
