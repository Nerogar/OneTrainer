from modules.trainer.EmbeddingTrainer import EmbeddingTrainer
from modules.trainer.FineTuneTrainer import FineTuneTrainer
from modules.trainer.LoraTrainer import LoraTrainer
from modules.util.args.TrainArgs import TrainArgs
from modules.util.enum.TrainingMethod import TrainingMethod


def main():
    args = TrainArgs.parse_args()

    trainer = None
    match args.training_method:
        case TrainingMethod.FINE_TUNE:
            trainer = FineTuneTrainer(args)
        case TrainingMethod.LORA:
            trainer = LoraTrainer(args)
        case TrainingMethod.EMBEDDING:
            trainer = EmbeddingTrainer(args)

    trainer.start()
    trainer.train()
    trainer.end()


if __name__ == '__main__':
    main()
