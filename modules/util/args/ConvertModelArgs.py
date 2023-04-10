import argparse

from modules.util.args.arg_type_util import *
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.ModelType import ModelType


class ConvertModelArgs:
    model_type: ModelType
    base_model_name: str
    output_dtype: torch.dtype
    output_model_format: ModelFormat
    output_model_destination: str

    def __init__(self, args: dict):
        for (key, value) in args.items():
            setattr(self, key, value)

    @staticmethod
    def parse_args() -> 'ConvertModelArgs':
        parser = argparse.ArgumentParser(description="One Trainer Training Script.")

        parser.add_argument("--model-type", type=ModelType, required=True, dest="model_type", help="Type of the base model", choices=list(ModelType))
        parser.add_argument("--base-model-name", type=str, required=True, dest="base_model_name", help="The base model to start training from")
        parser.add_argument("--output-dtype", type=torch_dtype, required=False, default="float16", dest="output_dtype", help="The data type to save the output model")
        parser.add_argument("--output-model-format", type=ModelFormat, required=False, default=ModelFormat.CKPT, dest="output_model_format", help="The format to save the final output model", choices=list(ModelFormat))
        parser.add_argument("--output-model-destination", type=str, required=True, dest="output_model_destination", help="The destination to save the final output model")

        return ConvertModelArgs(vars(parser.parse_args()))
