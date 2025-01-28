"""
The `sample.py` script is a tool for generating samples from a pre-trained model.

It uses `SampleArgs` for parsing command-line arguments to adjust sampling parameters.
The script employs `create_model_loader` to load the model and `create_model_setup` to prepare it for sampling.
`create_model_sampler` is then used to generate the sample.
This script is useful for quickly generating images.
"""
from util.import_util import script_imports

script_imports()

from modules.util import create
from modules.util.args.SampleArgs import SampleArgs
from modules.util.config.SampleConfig import SampleConfig
from modules.util.enum.ImageFormat import ImageFormat
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.ModelNames import ModelNames
from modules.util.torch_util import default_device


def main():
    """
    Samples from a model.

    Parses command line arguments using SampleArgs.
    Creates a model loader and setup based on the specified model type.
    Loads the base model and sets it to evaluation mode.
    Creates a model sampler based on the model type.
    Samples the model and saves the output to the specified destination.
    """
    args = SampleArgs.parse_args()
    device = default_device

    training_method = TrainingMethod.FINE_TUNE

    model_loader = create.create_model_loader(args.model_type, training_method=training_method)
    create.create_model_setup(args.model_type, device, device, training_method=training_method)

    print("Loading model " + args.base_model_name)
    model = model_loader.load(
        model_type=args.model_type,
        model_names=ModelNames(base_model=args.base_model_name),
        weight_dtypes=args.weight_dtypes(),
    )

    model.to(device)
    model.eval()

    model_sampler = create.create_model_sampler(
        train_device=device,
        temp_device=device,
        model=model,
        model_type=args.model_type,
    )

    print("Sampling " + args.destination)
    model_sampler.sample(
        sample_params=SampleConfig.default_values().from_dict(
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
