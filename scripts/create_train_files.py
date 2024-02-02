import os
import sys

sys.path.append(os.getcwd())

import json
from pathlib import Path

from modules.util.args.CreateTrainFilesArgs import CreateTrainFilesArgs
from modules.util.config.ConceptParams import ConceptConfig
from modules.util.config.SampleParams import SampleConfig

def main():
    args = CreateTrainFilesArgs.parse_args()

    if args.concepts_output_destination:
        data = [ConceptConfig.default_values().to_dict()]
        os.makedirs(Path(path=args.concepts_output_destination).parent.absolute(), exist_ok=True)

        with open(args.concepts_output_destination, "w") as f:
            json.dump(data, f, indent=4)

    if args.samples_output_destination:
        data = [SampleConfig.default_values().to_dict()]
        os.makedirs(Path(path=args.samples_output_destination).parent.absolute(), exist_ok=True)

        with open(args.samples_output_destination, "w") as f:
            json.dump(data, f, indent=4)


if __name__ == '__main__':
    main()
