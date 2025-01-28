"""
This script, `calculate_loss.py`, is a utility for evaluating model performance by calculating loss.

It leverages the `GenerateLossesModel` class, which encapsulates the logic for loss calculation.
The script requires a JSON configuration file (`config_path`) to specify training parameters, enabling flexible loss calculations for various training scenarios.
It plays a crucial role in the training pipeline, particularly when used in conjunction with `train.py`.
By calculating the model loss, it allows the trainer to determine the model performance.
"""
from util.import_util import script_imports

script_imports()

import json

from modules.module.GenerateLossesModel import GenerateLossesModel
from modules.util.args.CalculateLossArgs import CalculateLossArgs
from modules.util.config.TrainConfig import TrainConfig


def main():
    """
    Calculates the loss of a model.

    Reads a JSON configuration file to configure the training process.
    Parses command line arguments using CalculateLossArgs.
    Initializes and starts the GenerateLossesModel trainer.
    """
    args = CalculateLossArgs.parse_args()

    train_config = TrainConfig.default_values()
    with open(args.config_path, "r") as f:
        train_config.from_dict(json.load(f))

    trainer = GenerateLossesModel(train_config, args.output_path)
    trainer.start()


if __name__ == '__main__':
    main()
