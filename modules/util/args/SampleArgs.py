import argparse
from typing import Any

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
    destination: str
    text_encoder_layer_skip: int

    def __init__(self, data: list[(str, Any, type, bool)]):
        super(SampleArgs, self).__init__(data)

    def weight_dtypes(self) -> ModelWeightDtypes:
        return ModelWeightDtypes(
            self.weight_dtype,
            self.weight_dtype,
            self.weight_dtype,
            self.weight_dtype,
            self.weight_dtype,
        )

    @staticmethod
    def parse_args() -> 'SampleArgs':
        parser = argparse.ArgumentParser(description="One Trainer Sampling Script.")

        # @formatter:off

        parser.add_argument("--model-type", type=ModelType, required=True, dest="model_type", help="Type of the base model", choices=list(ModelType))
        parser.add_argument("--weight-dtype", type=DataType, required=False, default=DataType.FLOAT_32, dest="weight_dtype", help="The data type to use for weights during sampling", choices=list(DataType))
        parser.add_argument("--base-model-name", type=str, required=True, dest="base_model_name", help="The base model to sample from")
        parser.add_argument("--embedding-name", type=str, required=False, default="", dest="extra_model_name", help="An embedding to use during sampling")
        parser.add_argument("--prompt", type=str, required=True, dest="prompt", help="The prompt for sampling")
        parser.add_argument("--negative-prompt", type=str, required=False, default="", dest="negative_prompt", help="The negative prompt for sampling")
        parser.add_argument("--destination", type=str, required=True, dest="destination", help="The destination to save the output")
        parser.add_argument("--text-encoder-layer-skip", type=int, required=False, default=0, dest="text_encoder_layer_skip", help="Skip last layers of the text encoder")

        # @formatter:on

        args = SampleArgs.default_values()
        args.from_dict(vars(parser.parse_args()))
        return args

    @staticmethod
    def default_values() -> 'SampleArgs':
        data = []

        # name, default value, data type, nullable
        data.append(("model_type", TrainingMethod.FINE_TUNE, TrainingMethod, False))
        data.append(("weight_dtype", DataType.FLOAT_32, DataType))
        data.append(("base_model_name", "", str, False))
        data.append(("embedding_name", None, str, True))
        data.append(("prompt", "", str, False))
        data.append(("negative_prompt", "", str, False))
        data.append(("destination", "", str, False))
        data.append(("text_encoder_layer_skip", 0, int, False))

        return SampleArgs(data)
