from util.import_util import script_imports

script_imports()

from modules.util import create
from modules.util.args.SampleArgs import SampleArgs
from modules.util.config.SampleConfig import SampleConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.DataType import DataType
from modules.util.enum.EMAMode import EMAMode
from modules.util.enum.ImageFormat import ImageFormat
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.ModelNames import ModelNames
from modules.util.torch_util import default_device


def main():
    args = SampleArgs.parse_args()
    device = default_device

    training_method = TrainingMethod.FINE_TUNE
    train_config = TrainConfig.default_values()
    train_config.optimizer.optimizer = None
    train_config.ema = EMAMode.OFF

    model_loader = create.create_model_loader(args.model_type, training_method=training_method)
    model_setup = create.create_model_setup(args.model_type, device, device, training_method=training_method)

    print("Loading model " + args.base_model_name)
    weight_dtypes = args.weight_dtypes()
    weight_dtypes.vae = DataType.FLOAT_32
    model = model_loader.load(
        model_type=args.model_type,
        model_names=ModelNames(base_model=args.base_model_name),
        weight_dtypes=weight_dtypes
    )
    model.train_config = train_config

    model_setup.setup_model(model, train_config)

    model.to(device)
    model.eval()
    model.train_dtype = args.weight_dtype

    model_sampler = create.create_model_sampler(
        train_device=device,
        temp_device=device,
        model=model,
        model_type=args.model_type,
    )

    print("Sampling " + args.destination)
    model_sampler.sample(
        sample_config=SampleConfig.default_values().from_dict(
            {
                "prompt": args.prompt,
                "negative_prompt": args.negative_prompt,
                "height": args.height,
                "width": args.width,
                "seed": 42,
                "text_encoder_1_layer_skip": args.text_encoder_layer_skip,
                "text_encoder_2_layer_skip": args.text_encoder_layer_skip,
                "text_encoder_3_layer_skip": args.text_encoder_layer_skip,
                "sample_inpainting": args.sample_inpainting,
                "base_image_path": args.base_image_path,
                "mask_image_path": args.mask_image_path,
            }
        ),
        image_format=ImageFormat.JPG,
        destination=args.destination,
    )


if __name__ == '__main__':
    main()
