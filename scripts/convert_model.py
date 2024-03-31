from util.import_util import script_imports

script_imports()

from modules.util.ModelNames import ModelNames
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util import create
from modules.util.args.ConvertModelArgs import ConvertModelArgs


def main():
    args = ConvertModelArgs.parse_args()

    model_loader = create.create_model_loader(model_type=args.model_type, training_method=args.training_method)
    model_saver = create.create_model_saver(model_type=args.model_type, training_method=args.training_method)

    print("Loading model " + args.input_name)
    if args.training_method in [TrainingMethod.FINE_TUNE]:
        model = model_loader.load(
            model_type=args.model_type,
            model_names=ModelNames(
                base_model=args.input_name,
            ),
            weight_dtypes=args.weight_dtypes(),
        )
    elif args.training_method in [TrainingMethod.LORA, TrainingMethod.EMBEDDING]:
        model = model_loader.load(
            model_type=args.model_type,
            model_names=ModelNames(
                lora=args.input_name,
                embedding=[args.input_name],
            ),
            weight_dtypes=args.weight_dtypes(),
        )
    else:
        raise Exception("could not load model: " + args.input_name)

    print("Saving model " + args.output_model_destination)
    model_saver.save(
        model=model,
        model_type=args.model_type,
        output_model_format=args.output_model_format,
        output_model_destination=args.output_model_destination,
        dtype=args.output_dtype.torch_dtype(),
    )


if __name__ == '__main__':
    main()
