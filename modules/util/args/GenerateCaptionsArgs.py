import argparse

from modules.util.enum.DataType import DataType
from modules.util.enum.GenerateCaptionsModel import GenerateCaptionsModel


class GenerateCaptionsArgs:
    model: GenerateCaptionsModel
    sample_dir: str
    initial_caption: str
    mode: str
    device: str
    dtype: DataType

    def __init__(self, args: dict):
        for (key, value) in args.items():
            setattr(self, key, value)

    @staticmethod
    def parse_args() -> 'GenerateCaptionsArgs':
        parser = argparse.ArgumentParser(description="One Trainer Generate Captions Script.")

        parser.add_argument("--model", type=GenerateCaptionsModel, required=True, dest="model", help="The model to use when generating captions")
        parser.add_argument("--sample-dir", type=str, required=True, dest="sample_dir", help="Directory where samples are located")
        parser.add_argument("--initial-caption", type=str, default='', required=False, dest="initial_caption", help="An initial caption to start generating from")
        parser.add_argument("--mode", type=str, default='fill', required=False, dest="mode", help="Either replace, fill, add or subtract")
        parser.add_argument("--device", type=str, required=False, default="cuda", dest="device", help="The device to use for calculations")
        parser.add_argument("--dtype", type=DataType, required=False, default=DataType.FLOAT_16, dest="dtype", help="The data type to use for weights during calculations", choices=list(DataType))

        return GenerateCaptionsArgs(vars(parser.parse_args()))
