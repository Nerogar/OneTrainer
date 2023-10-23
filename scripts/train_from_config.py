import json
import os
import sys

sys.path.append(os.getcwd())

from modules.util.args.TrainArgs import TrainArgs
from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.args.TrainFromConfigArgs import TrainFromConfigArgs

from modules.trainer.GenericTrainer import GenericTrainer


def main():
    args = TrainFromConfigArgs.parse_args()
    callbacks = TrainCallbacks()
    commands = TrainCommands()

    train_args = TrainArgs.default_values()

    with open(args.config_path, "r") as f:
        train_args.from_dict(json.load(f))

    trainer = GenericTrainer(train_args, callbacks, commands)

    trainer.start()

    canceled = False
    try:
        trainer.train()
    except KeyboardInterrupt:
        canceled = True

    if not canceled or train_args.backup_before_save:
        trainer.end()


if __name__ == '__main__':
    main()
