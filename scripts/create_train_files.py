import os
import sys

sys.path.append(os.getcwd())

import json
from pathlib import Path

from modules.util.args import concept_defaults, sample_defaults
from modules.util.args.CreateTrainFilesArgs import CreateTrainFilesArgs


def main():
    args = CreateTrainFilesArgs.parse_args()

    if args.concepts_output_destination:
        data = [concept_defaults.create_new_concept()]
        os.makedirs(Path(path=args.concepts_output_destination).parent.absolute(), exist_ok=True)

        with open(args.concepts_output_destination, "w") as f:
            json.dump(data, f, indent=4)

    if args.samples_output_destination:
        data = [sample_defaults.create_new_sample()]
        os.makedirs(Path(path=args.samples_output_destination).parent.absolute(), exist_ok=True)

        with open(args.samples_output_destination, "w") as f:
            json.dump(data, f, indent=4)


if __name__ == '__main__':
    main()
