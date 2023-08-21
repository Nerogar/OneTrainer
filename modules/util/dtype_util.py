from modules.util.args.TrainArgs import TrainArgs
from modules.util.enum.DataType import DataType


def allow_mixed_precision(train_args: TrainArgs):
    all_dtypes = list(train_args.weight_dtypes().all_dtypes() + [train_args.train_dtype])
    all_dtypes = list(filter(lambda dtype: dtype != DataType.NONE, all_dtypes))
    all_dtypes = set(all_dtypes)

    return len(all_dtypes) != 1
