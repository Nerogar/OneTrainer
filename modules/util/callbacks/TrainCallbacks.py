from typing import Callable

from PIL.Image import Image

from modules.util.TrainProgress import TrainProgress


class TrainCallbacks:
    def __init__(
            self,
            on_update_progress: Callable[[TrainProgress, int, int], None] = lambda _: None,
            on_update_status: Callable[[str], None] = lambda _: None,
            on_sample: Callable[[Image], None] = lambda _: None,
    ):
        self.__on_update_progress = on_update_progress
        self.__on_update_status = on_update_status
        self.__on_sample = on_sample

    def on_update_progress(self, train_progress: TrainProgress, max_sample: int, max_epoch: int):
        try:
            if self.__on_update_progress:
                self.__on_update_progress(train_progress, max_sample, max_epoch)
        except:
            pass

    def on_update_status(self, status: str):
        try:
            if self.__on_update_status:
                self.__on_update_status(status)
        except:
            pass

    def on_sample(self, sample: Image):
        try:
            if self.__on_sample:
                self.__on_sample(sample)
        except:
            pass
