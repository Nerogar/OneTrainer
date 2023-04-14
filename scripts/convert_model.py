import os
import sys

sys.path.append(os.getcwd())

from modules.util import create
from modules.util.args.ConvertModelArgs import ConvertModelArgs


def main():
    args = ConvertModelArgs.parse_args()

    model_loader = create.create_model_loader(args.model_type)
    model_saver = create.create_model_saver(args.model_type)

    print("Loading model " + args.base_model_name)
    model = model_loader.load(args.base_model_name, args.model_type)

    print("Saving model " + args.output_model_destination)
    model_saver.save(model, args.model_type, args.output_model_format, args.output_model_destination, args.output_dtype)


if __name__ == '__main__':
    main()
