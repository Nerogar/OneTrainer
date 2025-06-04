from util.import_util import script_imports

script_imports()

from modules.module.BaseBooruModel import JoyTagBooruModel, WDBooruModel
from modules.module.Blip2Model import Blip2Model
from modules.module.MoondreamModel import MoondreamModel
from modules.util.args.GenerateCaptionsArgs import GenerateCaptionsArgs
from modules.util.enum.GenerateCaptionsModel import GenerateCaptionsModel

import torch


def main():
    args = GenerateCaptionsArgs.parse_args()

    model = None
    if args.model == GenerateCaptionsModel.MOONDREAM2:
        model = MoondreamModel(torch.device(args.device), args.dtype.torch_dtype())
    elif args.model == GenerateCaptionsModel.BLIP2:
        model = Blip2Model(torch.device(args.device), args.dtype.torch_dtype())
    elif args.model == GenerateCaptionsModel.WD14_VIT_2:
        model = WDBooruModel(torch.device(args.device), args.dtype.torch_dtype(), model_name="WD14 VIT v2")
    elif args.model == GenerateCaptionsModel.WD_EVA02_LARGE_V3:
        model = WDBooruModel(torch.device(args.device), args.dtype.torch_dtype(), model_name="WD EVA02-Large Tagger v3")
    elif args.model == GenerateCaptionsModel.WD_SWINV2_V3:
        model = WDBooruModel(torch.device(args.device), args.dtype.torch_dtype(), model_name="WD SwinV2 Tagger v3")
    elif args.model == GenerateCaptionsModel.JOYTAG:
        model = JoyTagBooruModel(torch.device(args.device), args.dtype.torch_dtype())

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
