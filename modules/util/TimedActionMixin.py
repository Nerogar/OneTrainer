import time

from modules.util.TrainProgress import TrainProgress
from modules.util.enum.TimeUnit import TimeUnit


class TimedActionMixin:
    def __init__(self):
        super(TimedActionMixin, self).__init__()
        self.__previous_action = {}
        self.__start_time = time.time()

    def repeating_action_needed(
            self,
            name: str,
            interval: float,
            unit: TimeUnit,
            train_progress: TrainProgress,
            start_at_zero: bool = True,
    ):
        if name not in self.__previous_action:
            self.__previous_action[name] = -1

        match unit:
            case TimeUnit.EPOCH:
                if start_at_zero:
                    return train_progress.epoch % int(interval) == 0 and train_progress.epoch_step == 0
                else:
                    # should actually be the last step of each epoch, but we don't know how many steps an epoch has
                    return train_progress.epoch % int(interval) == 0 and train_progress.epoch_step == 0 \
                        and train_progress.epoch > 0
            case TimeUnit.STEP:
                if start_at_zero:
                    return train_progress.global_step % int(interval) == 0
                else:
                    return (train_progress.global_step + 1) % int(interval) == 0
            case TimeUnit.SECOND:
                if not start_at_zero and self.__previous_action[name] < 0:
                    self.__previous_action[name] = time.time()

                seconds_since_previous_action = time.time() - self.__previous_action[name]
                if seconds_since_previous_action > interval:
                    self.__previous_action[name] = time.time()
                    return True
                else:
                    return False
            case TimeUnit.MINUTE:
                if not start_at_zero and self.__previous_action[name] < 0:
                    self.__previous_action[name] = time.time()

                seconds_since_previous_action = time.time() - self.__previous_action[name]
                if seconds_since_previous_action > (interval * 60):
                    self.__previous_action[name] = time.time()
                    return True
                else:
                    return False
            case TimeUnit.HOUR:
                if not start_at_zero and self.__previous_action[name] < 0:
                    self.__previous_action[name] = time.time()

                seconds_since_previous_action = time.time() - self.__previous_action[name]
                if seconds_since_previous_action > (interval * 60 * 60):
                    self.__previous_action[name] = time.time()
                    return True
                else:
                    return False
            case TimeUnit.NEVER:
                return False
            case TimeUnit.ALWAYS:
                return True
            case _:
                return False

    def single_action_elapsed(
            self,
            name: str,
            delay: float,
            unit: TimeUnit,
            train_progress: TrainProgress,
    ):
        if name not in self.__previous_action:
            self.__previous_action[name] = time.time()

        match unit:
            case TimeUnit.EPOCH:
                return train_progress.epoch > int(delay)
            case TimeUnit.STEP:
                return train_progress.global_step > int(delay)
            case TimeUnit.SECOND:
                seconds_since_start = time.time() - self.__start_time
                return seconds_since_start > delay
            case TimeUnit.MINUTE:
                seconds_since_start = time.time() - self.__start_time
                return seconds_since_start > (delay * 60)
            case TimeUnit.HOUR:
                seconds_since_start = time.time() - self.__start_time
                return seconds_since_start > (delay * 60 * 60)
            case TimeUnit.NEVER:
                return False
            case TimeUnit.ALWAYS:
                return True
            case _:
                return False
