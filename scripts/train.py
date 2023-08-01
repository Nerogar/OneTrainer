import os
import sys

sys.path.append(os.getcwd())

from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from modules.util.commands.TrainCommands import TrainCommands

from modules.trainer.GenericTrainer import GenericTrainer
from modules.util.args.TrainArgs import TrainArgs


def main():
    args = TrainArgs.parse_args()
    callbacks = TrainCallbacks()
    commands = TrainCommands()

    trainer = GenericTrainer(args, callbacks, commands)

    trainer.start()

    canceled = False
    try:
        trainer.train()
    except KeyboardInterrupt:
        canceled = True

    if not canceled or args.backup_before_save:
        trainer.end()


if __name__ == '__main__':
    main()
