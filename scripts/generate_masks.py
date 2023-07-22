import os
import sys

sys.path.append(os.getcwd())

from modules.util.args.GenerateMasksArgs import GenerateMasksArgs
from modules.module.ClipSegModel import ClipSegModel


def main():
    args = GenerateMasksArgs.parse_args()
    clip_seg = ClipSegModel()
    clip_seg.mask_folder(
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
