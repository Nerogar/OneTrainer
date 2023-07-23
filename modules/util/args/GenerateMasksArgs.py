import argparse

from modules.util.enum.DataType import DataType
from modules.util.enum.GenerateMasksModel import GenerateMasksModel


class GenerateMasksArgs:
    model: GenerateMasksModel
    sample_dir: str
    prompts: list[str]
    mode: str
    threshold: float
    smooth_pixels: int
    expand_pixels: int
    device: str
    dtype: DataType

    def __init__(self, args: dict):
        for (key, value) in args.items():
            setattr(self, key, value)

    @staticmethod
    def parse_args() -> 'GenerateMasksArgs':
        parser = argparse.ArgumentParser(description="One Trainer Generate Masks Script.")

        parser.add_argument("--model", type=GenerateMasksModel, required=True, dest="model", help="The model to use when generating masks")
        parser.add_argument("--sample-dir", type=str, required=True, dest="sample_dir", help="Directory where samples are located")
        parser.add_argument("--add-prompt", type=str, required=True, action="append", dest="prompts", help="A prompt used to create a mask")
        parser.add_argument("--mode", type=str, default='fill', required=False, dest="mode", help="Either replace, fill, add or subtract")
        parser.add_argument("--threshold", type=float, default='0.3', required=False, dest="threshold", help="Threshold for including pixels in the mask")
        parser.add_argument("--smooth-pixels", type=int, default=5, required=False, dest="smooth_pixels", help="Radius of a smoothing operation applied to the generated mask")
        parser.add_argument("--expand-pixels", type=int, default=10, required=False, dest="expand_pixels", help="Amount of expansion of the generated mask in all directions")
        parser.add_argument("--device", type=str, required=False, default="cuda", dest="device", help="The device to use for calculations")
        parser.add_argument("--dtype", type=DataType, required=False, default=DataType.FLOAT_32, dest="dtype", help="The data type to use for weights during calculations", choices=list(DataType))

        return GenerateMasksArgs(vars(parser.parse_args()))
