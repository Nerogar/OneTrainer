"""
The `generate_captions.py` script automates the process of generating textual descriptions for images.

It supports multiple captioning models, including `Blip2Model`, `BlipModel`, and `WDModel`.
The script uses `GenerateCaptionsArgs` for parsing command-line arguments, providing a flexible interface for various options.
This script is typically employed to produce captions for datasets used in training.
"""
from util.import_util import script_imports

script_imports()

from modules.module.Blip2Model import Blip2Model
from modules.module.BlipModel import BlipModel
from modules.module.WDModel import WDModel
from modules.util.args.GenerateCaptionsArgs import GenerateCaptionsArgs
from modules.util.enum.GenerateCaptionsModel import GenerateCaptionsModel

import torch


def main():
    """
    Generates captions for images.

    Parses command line arguments using GenerateCaptionsArgs.
    Initializes the specified model for generating captions.
    Captions the folder of images using the specified model.
    Prints an error message if there is an error while processing the image.
    """
    args = GenerateCaptionsArgs.parse_args()

    model = None
    if args.model == GenerateCaptionsModel.BLIP:
        model = BlipModel(torch.device(args.device), args.dtype.torch_dtype())
    elif args.model == GenerateCaptionsModel.BLIP2:
        model = Blip2Model(torch.device(args.device), args.dtype.torch_dtype())
    elif args.model == GenerateCaptionsModel.WD14_VIT_2:
        model = WDModel(torch.device(args.device), args.dtype.torch_dtype())

    model.caption_folder(
        sample_dir=args.sample_dir,
        initial_caption=args.initial_caption,
        caption_prefix=args.caption_prefix,
        caption_postfix=args.caption_postfix,
        mode=args.mode,
        error_callback=lambda filename: print("Error while processing image " + filename),
        include_subdirectories=args.include_subdirectories
    )


if __name__ == "__main__":
    main()
