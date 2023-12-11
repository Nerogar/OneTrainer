import os
import sys
import json

sys.path.append(os.getcwd())

from modules.util.args.TrainArgs import TrainArgs
from modules.util.args.TrainFromConfigArgs import TrainFromConfigArgs

from modules.util.enum.GenerateLossesModel import GenerateLossesModel


def main():
    args = TrainFromConfigArgs.parse_args()
    with open(args.config_path) as f:
        train_args = TrainArgs.from_dict(json.load(f))
    trainer = GenerateLossesModel(train_args)
    trainer.start()

if __name__ == '__main__':
    main()
