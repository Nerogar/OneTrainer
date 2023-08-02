from modules.util.args.TrainArgs import TrainArgs
from modules.util.enum.TrainingMethod import TrainingMethod


def allow_mixed_precision(train_args: TrainArgs):
    return any([
        train_args.train_dtype.enable_mixed_precision(train_args.weight_dtype),
        train_args.training_method == TrainingMethod.LORA and train_args.weight_dtype != train_args.lora_weight_dtype,
    ])
