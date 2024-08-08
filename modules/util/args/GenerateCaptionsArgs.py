import argparse
from typing import Any

from modules.util.args.BaseArgs import BaseArgs
from modules.util.enum.DataType import DataType
from modules.util.enum.GenerateCaptionsModel import GenerateCaptionsModel
from modules.util.torch_util import default_device


class GenerateCaptionsArgs(BaseArgs):
    model: GenerateCaptionsModel
    sample_dir: str
    initial_caption: str
    caption_prefix: str
    caption_postfix: str
    mode: str
    device: str
    dtype: DataType
    include_subdirectories: bool

    def __init__(self, data: list[(str, Any, type, bool)]):
        super().__init__(data)

    @staticmethod
    def parse_args() -> 'GenerateCaptionsArgs':
        parser = argparse.ArgumentParser(description="One Trainer Generate Captions Script.")

        # @formatter:off

        parser.add_argument("--model", type=GenerateCaptionsModel, required=True, dest="model", help="The model to use when generating captions", choices=list(GenerateCaptionsModel))
        parser.add_argument("--sample-dir", type=str, required=True, dest="sample_dir", help="Directory where samples are located")
        parser.add_argument("--initial-caption", type=str, default='', required=False, dest="initial_caption", help="An initial caption to start generating from")
        parser.add_argument("--caption-prefix", type=str, default='', required=False, dest="caption_prefix", help="Add this to the start of the generated caption (before initial caption)")
        parser.add_argument("--caption-postfix", type=str, default='', required=False, dest="caption_postfix", help="Add this to the end of the generated caption")
        parser.add_argument("--mode", type=str, default='fill', required=False, dest="mode", help="Either replace, fill or add")
        parser.add_argument("--device", type=str, required=False, default=default_device.type, dest="device", help="The device to use for calculations")
        parser.add_argument("--dtype", type=DataType, required=False, default=DataType.FLOAT_16, dest="dtype", help="The data type to use for weights during calculations", choices=list(DataType))
        parser.add_argument("--include-subdirectories", action="store_true", required=False, default=False, dest="include_subdirectories", help="Whether to include subdirectories when processing samples")

        # @formatter:on

        args = GenerateCaptionsArgs.default_values()
        args.from_dict(vars(parser.parse_args()))
        return args

    @staticmethod
    def default_values():
        data = []

        data.append(("model", GenerateCaptionsModel.BLIP, GenerateCaptionsModel, False))
        data.append(("sample_dir", "", str, False))
        data.append(("initial_caption", "", str, False))
        data.append(("caption_prefix", "", str, False))
        data.append(("caption_postfix", "", str, False))
        data.append(("mode", "fill", str, False))
        data.append(("device", default_device.type, str, False))
        data.append(("dtype", DataType.FLOAT_16, DataType, False))
        data.append(("include_subdirectories", False, bool, False))

        return GenerateCaptionsArgs(data)
