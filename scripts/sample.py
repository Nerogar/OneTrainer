from util.import_util import script_imports

script_imports()

from modules.util.config.SampleConfig import SampleConfig
from modules.util.enum.ImageFormat import ImageFormat
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util import create
from modules.util.args.SampleArgs import SampleArgs
from modules.util.torch_util import default_device


def main():
    args = SampleArgs.parse_args()
    device = default_device

    training_method = TrainingMethod.FINE_TUNE
    if args.embedding_name is not None:
        training_method = TrainingMethod.EMBEDDING

    model_loader = create.create_model_loader(args.model_type, training_method=training_method)
    model_setup = create.create_model_setup(args.model_type, device, device, training_method=training_method)

    print("Loading model " + args.base_model_name)
    model = model_loader.load(
        model_type=args.model_type,

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
                "height": 512,
                "width": 512,
                "seed": 42,
            }
        ),
        image_format=ImageFormat.JPG,
        destination=args.destination,
        text_encoder_layer_skip=args.text_encoder_layer_skip,
    )


if __name__ == '__main__':
    main()
