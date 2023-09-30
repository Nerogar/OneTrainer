import os
import sys

sys.path.append(os.getcwd())

import torch
from modules.util.enum.GenerateMasksModel import GenerateMasksModel
from modules.util.args.GenerateMasksArgs import GenerateMasksArgs
from modules.module.RembgModel import RembgModel
from modules.module.ClipSegModel import ClipSegModel
from modules.module.MaskByColor import MaskByColor


def main():
    args = GenerateMasksArgs.parse_args()

    model = None
    if args.model == GenerateMasksModel.CLIPSEG:
        model = ClipSegModel(torch.device(args.device), args.dtype.torch_dtype())
    elif args.model == GenerateMasksModel.REMBG:
        model = RembgModel(torch.device(args.device), args.dtype.torch_dtype())
    elif args.model == GenerateMasksModel.COLOR:
        model = MaskByColor(torch.device(args.device), args.dtype.torch_dtype())

    model.mask_folder(
        sample_dir=args.sample_dir,
        prompts=args.prompts,
        mode=args.mode,
        threshold=args.threshold,
        smooth_pixels=args.smooth_pixels,
        expand_pixels=args.expand_pixels,
        error_callback=lambda filename: print("Error while processing image " + filename)
    )


if __name__ == "__main__":
    main()
