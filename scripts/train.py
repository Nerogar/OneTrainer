import os
import sys

from modules.util.callbacks.TrainCallbacks import TrainCallbacks

sys.path.append(os.getcwd())

from modules.trainer.FineTuneTrainer import FineTuneTrainer
from modules.util.args.TrainArgs import TrainArgs
from modules.util.enum.TrainingMethod import TrainingMethod


def main():
    args = TrainArgs.parse_args()
    callbacks = TrainCallbacks()

    trainer = None
    match args.training_method:
        case TrainingMethod.FINE_TUNE:
            trainer = FineTuneTrainer(args, callbacks)
        case TrainingMethod.LORA:
            trainer = FineTuneTrainer(args, callbacks)
        case TrainingMethod.EMBEDDING:
            trainer = FineTuneTrainer(args, callbacks)
        case TrainingMethod.FINE_TUNE_VAE:
            trainer = FineTuneTrainer(args, callbacks)

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
