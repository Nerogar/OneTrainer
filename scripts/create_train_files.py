"""
`create_train_files.py` is designed to generate essential configuration and data files for training.

It processes command-line arguments via `CreateTrainFilesArgs`, providing flexibility in specifying output destinations.
The script constructs default configurations for `TrainConfig`, `ConceptConfig`, and `SampleConfig`, which are then serialized into JSON format.
The generated files are used by `train.py` to train a model.
"""
from util.import_util import script_imports

script_imports()

import json
import os
from pathlib import Path

from modules.util.args.CreateTrainFilesArgs import CreateTrainFilesArgs
from modules.util.config.ConceptConfig import ConceptConfig
from modules.util.config.SampleConfig import SampleConfig
from modules.util.config.TrainConfig import TrainConfig


def main():
    """
    Creates training files (config, concepts, samples).

    Parses command line arguments using CreateTrainFilesArgs.
    Creates default configuration, concept, and sample files if output destinations are specified.
    Writes the created files to the specified output destinations in JSON format.
    """
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
