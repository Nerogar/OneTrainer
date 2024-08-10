from util.import_util import script_imports

script_imports()

import json

from modules.util.config.TrainConfig import TrainConfig
from modules.module.GenerateLossesModel import GenerateLossesModel
from modules.util.args.CalculateLossArgs import CalculateLossArgs


def main():
    args = CalculateLossArgs.parse_args()

    train_config = TrainConfig.default_values()
    with open(args.config_path, "r") as f:
        train_config.from_dict(json.load(f))

    trainer = GenerateLossesModel(train_config, args.output_path)
    trainer.start()


if __name__ == '__main__':
    main()
