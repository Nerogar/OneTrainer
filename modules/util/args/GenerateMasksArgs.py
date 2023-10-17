import argparse
from typing import Any

from modules.util.args.BaseArgs import BaseArgs
from modules.util.enum.DataType import DataType
from modules.util.enum.GenerateMasksModel import GenerateMasksModel


class GenerateMasksArgs(BaseArgs):
    model: GenerateMasksModel
    sample_dir: str
    prompts: list[str]
    mode: str
    threshold: float
    smooth_pixels: int
    expand_pixels: int
    device: str
    dtype: DataType
    alpha: float | None

    def __init__(self, data: list[(str, Any, type, bool)]):
        super(GenerateMasksArgs, self).__init__(data)

    @staticmethod
    def parse_args() -> 'GenerateMasksArgs':
        parser = argparse.ArgumentParser(description="One Trainer Generate Masks Script.")

        # @formatter:off

        parser.add_argument("--model", type=GenerateMasksModel, required=True, dest="model", help="The model to use when generating masks")
        parser.add_argument("--sample-dir", type=str, required=True, dest="sample_dir", help="Directory where samples are located")
        parser.add_argument("--add-prompt", type=str, required=True, action="append", dest="prompts", help="A prompt used to create a mask")
        parser.add_argument("--mode", type=str, default='fill', required=False, dest="mode", help="Either replace, fill, add or subtract")
        parser.add_argument("--threshold", type=float, default=0.3, required=False, dest="threshold", help="Threshold for including pixels in the mask")
        parser.add_argument("--smooth-pixels", type=int, default=5, required=False, dest="smooth_pixels", help="Radius of a smoothing operation applied to the generated mask")
        parser.add_argument("--expand-pixels", type=int, default=10, required=False, dest="expand_pixels", help="Amount of expansion of the generated mask in all directions")
        parser.add_argument("--device", type=str, required=False, default="cuda", dest="device", help="The device to use for calculations")
        parser.add_argument("--dtype", type=DataType, required=False, default=DataType.FLOAT_32, dest="dtype", help="The data type to use for weights during calculations", choices=list(DataType))
        parser.add_argument("--alpha", type=float, required=False, default=None, dest="alpha", help="The ratio to use when doing addition/subtraction. Default is to add/subtract images completely.")

        # @formatter:on

        args = GenerateMasksArgs.default_values()
        args.from_dict(vars(parser.parse_args()))
        return args

    @staticmethod
    def default_values():
        data = []

        data.append(("model", GenerateMasksModel.CLIPSEG, GenerateMasksModel, False))
        data.append(("sample_dir", "", str, False))
        data.append(("prompts", [], list[str], False))
        data.append(("mode", "fill", str, False))
        data.append(("threshold", 0.3, float, False))
        data.append(("smooth_pixels", 5, int, False))
        data.append(("expand_pixels", 10, int, False))
        data.append(("device", "cuda", str, False))
        data.append(("dtype", DataType.FLOAT_16, DataType, False))
        data.append(("alpha", None, float, True))

        return GenerateMasksArgs(data)
