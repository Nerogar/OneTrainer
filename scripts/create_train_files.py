from util.import_util import script_imports

script_imports()

import json

from pathlib import Path
from modules.util.args.CreateTrainFilesArgs import CreateTrainFilesArgs
from modules.util.config.ConceptConfig import ConceptConfig
from modules.util.config.SampleConfig import SampleConfig
from modules.util.config.TrainConfig import TrainConfig


def main():
    args = CreateTrainFilesArgs.parse_args()

    print(args.to_dict())

    if args.config_output_destination:
        print("config")
        data = TrainConfig.default_values().to_dict()
        os.makedirs(Path(path=args.config_output_destination).parent.absolute(), exist_ok=True)

        with open(args.config_output_destination, "w") as f:
            json.dump(data, f, indent=4)

    if args.concepts_output_destination:
        print("concepts")
        data = [ConceptConfig.default_values().to_dict()]
        os.makedirs(Path(path=args.concepts_output_destination).parent.absolute(), exist_ok=True)

        with open(args.concepts_output_destination, "w") as f:
            json.dump(data, f, indent=4)

    if args.samples_output_destination:
        print("samples")
        data = [SampleConfig.default_values().to_dict()]
        os.makedirs(Path(path=args.samples_output_destination).parent.absolute(), exist_ok=True)

        with open(args.samples_output_destination, "w") as f:
            json.dump(data, f, indent=4)


if __name__ == '__main__':
    main()
