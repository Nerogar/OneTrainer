import os
import sys

sys.path.append(os.getcwd())

from modules.util.args.GenerateCaptionsArgs import GenerateCaptionsArgs
from modules.module.BlipModel import BlipModel


def main():
    args = GenerateCaptionsArgs.parse_args()
    clip_seg = BlipModel()
    clip_seg.caption_folder(
        sample_dir=args.sample_dir,
        mode=args.mode,
        error_callback=lambda filename: print("Error while processing image " + filename)
    )


if __name__ == "__main__":
    main()
