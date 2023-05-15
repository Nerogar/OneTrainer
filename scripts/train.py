import os
import sys

from modules.util.callbacks.TrainCallbacks import TrainCallbacks

sys.path.append(os.getcwd())

from modules.trainer.GenericTrainer import GenericTrainer
from modules.util.args.TrainArgs import TrainArgs


def main():
    args = TrainArgs.parse_args()
    callbacks = TrainCallbacks()

    trainer = GenericTrainer(args, callbacks)

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
