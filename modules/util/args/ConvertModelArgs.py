import argparse

from modules.util.args.BaseArgs import BaseArgs
from modules.util.enum.DataType import DataType
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod


class ConvertModelArgs(BaseArgs):
    model_type: ModelType
    training_method: TrainingMethod
    input_name: str
    output_dtype: DataType
    output_model_format: ModelFormat
    output_model_destination: str

    def __init__(self, args: dict):
        super(ConvertModelArgs, self).__init__(args)

    @staticmethod
    def parse_args() -> 'ConvertModelArgs':
        parser = argparse.ArgumentParser(description="One Trainer Converter Script.")

        parser.add_argument("--model-type", type=ModelType, required=True, dest="model_type", help="Type of the base model", choices=list(ModelType))
        parser.add_argument("--training-method", type=TrainingMethod, required=False, default=TrainingMethod.FINE_TUNE, dest="training_method", help="The training method", choices=list(TrainingMethod))
        parser.add_argument("--input-name", type=str, required=True, dest="input_name", help="The model to convert")
        parser.add_argument("--output-dtype", type=DataType, required=False, default=DataType.FLOAT_16, dest="output_dtype", help="The data type to save the output model", choices=list(DataType))
        parser.add_argument("--output-model-format", type=ModelFormat, required=False, default=ModelFormat.SAFETENSORS, dest="output_model_format", help="The format to save the final output model", choices=list(ModelFormat))
        parser.add_argument("--output-model-destination", type=str, required=True, dest="output_model_destination", help="The destination to save the final output model")

        return ConvertModelArgs(vars(parser.parse_args()))

    @staticmethod
    def default_values():
        args = {}

        args["model_type"] = ModelType.STABLE_DIFFUSION_15
        args["training_method"] = TrainingMethod.FINE_TUNE
        args["input_name"] = ""
        args["output_dtype"] = DataType.FLOAT_16
        args["output_model_format"] = ModelFormat.SAFETENSORS
        args["output_model_destination"] = ""

        return ConvertModelArgs(args)
