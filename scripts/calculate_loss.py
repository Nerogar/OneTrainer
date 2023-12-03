import os
import sys

sys.path.append(os.getcwd())

from modules.util.args.TrainArgs import TrainArgs
from modules.util.enum.GenerateLossesModel import GenerateLossesModel


def main():
    args = TrainArgs.parse_args()
    trainer = GenerateLossesModel(args)
    trainer.start()

if __name__ == '__main__':
    main()
