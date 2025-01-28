"""
The `convert_model.py` script provides functionality for model conversion between different types and formats.

It leverages the `create_model_loader` and `create_model_saver` utilities to manage the loading and saving of models.
This script is essential when transitioning a model from one representation (e.g., a training checkpoint) to another (e.g., a deployable format).
It accepts `ConvertModelArgs` to parse command-line inputs for flexibility.
It supports various `TrainingMethod` enums, such as FINE_TUNE, LORA, and EMBEDDING.
It outputs a new model in the desired format, and dtype.
"""
from util.import_util import script_imports

script_imports()

from uuid import uuid4

from modules.util import create
from modules.util.args.ConvertModelArgs import ConvertModelArgs
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.ModelNames import EmbeddingName, ModelNames


def main():
    """
    Converts a model from one type to another.

    Parses command line arguments using ConvertModelArgs.
    Creates a model loader and saver based on the specified model type and training method.
    Loads the input model and saves it to the output destination in the specified format and data type.

    Raises:
        Exception: If the model cannot be loaded due to an unknown training method.
    """
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
                embedding=EmbeddingName(str(uuid4()), args.input_name),
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
