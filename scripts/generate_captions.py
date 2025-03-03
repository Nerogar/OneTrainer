from util.import_util import script_imports

script_imports()

from modules.module.Blip2Model import Blip2Model
from modules.module.BlipModel import BlipModel
from modules.module.WDModel import WDModel
from modules.util.args.GenerateCaptionsArgs import GenerateCaptionsArgs
from modules.util.enum.GenerateCaptionsModel import GenerateCaptionsModel

import torch


def main():
    args = GenerateCaptionsArgs.parse_args()

    model = None
    if args.model == GenerateCaptionsModel.BLIP:
        model = BlipModel(torch.device(args.device), args.dtype.torch_dtype())
    elif args.model == GenerateCaptionsModel.BLIP2:
        model = Blip2Model(torch.device(args.device), args.dtype.torch_dtype())
    elif args.model == GenerateCaptionsModel.WD14_VIT_2:
        model = WDModel(torch.device(args.device), args.dtype.torch_dtype(), "wd-v1-4-vit-tagger-v2", 0.35)
    elif args.model == GenerateCaptionsModel.WD14_SWINV2_v3:
        model = WDModel(torch.device(args.device), args.dtype.torch_dtype(), "wd-swinv2-tagger-v3", 0.35)
    elif args.model == GenerateCaptionsModel.WD14_EVA02_v3:
        model = WDModel(torch.device(args.device), args.dtype.torch_dtype(), "wd-eva02-large-tagger-v3", 0.5)

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
