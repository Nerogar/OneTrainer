from modules.util import create
from modules.util.args.ConvertModelArgs import ConvertModelArgs


def main():
    args = ConvertModelArgs.parse_args()

    model_loader = create.create_model_loader()
    model_saver = create.create_model_saver()

    print("Loading model " + args.base_model_name)
    model = model_loader.load(args.base_model_name, args.model_type, args.output_dtype)

    print("Saving model " + args.output_model_destination)
    model_saver.save(model, args.model_type, args.output_model_format, args.output_model_destination)


if __name__ == '__main__':
    main()
