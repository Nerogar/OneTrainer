from typing import Callable

from PIL.Image import Image

from modules.util.TrainProgress import TrainProgress


class TrainCallbacks:
    def __init__(
            self,
            on_update_progress: Callable[[TrainProgress, int, int], None] = lambda _: None,
            on_update_status: Callable[[str], None] = lambda _: None,
            on_sample_default: Callable[[Image], None] = lambda _: None,
            on_sample_custom: Callable[[Image], None] = lambda _: None,
    ):
        self.__on_update_progress = on_update_progress
        self.__on_update_status = on_update_status
        self.__on_sample_default = on_sample_default
        self.__on_sample_custom = on_sample_custom

    def set_on_update_progress(
            self,
            on_update_progress: Callable[[TrainProgress, int, int], None] = lambda _: None,
    ):
        self.__on_update_progress = on_update_progress

    def on_update_progress(self, train_progress: TrainProgress, max_sample: int, max_epoch: int):
        try:
            if self.__on_update_progress:
                self.__on_update_progress(train_progress, max_sample, max_epoch)
        except:
            pass

    def set_on_update_status(
            self,
            on_update_status: Callable[[str], None] = lambda _: None,
    ):
        self.__on_update_status = on_update_status

    def on_update_status(self, status: str):
        try:
            if self.__on_update_status:
                self.__on_update_status(status)
        except:
            pass

    def set_on_sample_default(
            self,
            on_sample_default: Callable[[Image], None] = lambda _: None,
    ):
        self.__on_sample_default = on_sample_default

    def on_sample_default(self, sample: Image):
        try:
            if self.__on_sample_default:
                self.__on_sample_default(sample)
        except:
            pass

    def set_on_sample_custom(
            self,
            on_sample_custom: Callable[[Image], None] = lambda _: None,
    ):
        self.__on_sample_custom = on_sample_custom

    def on_sample_custom(self, sample: Image):
        try:
            if self.__on_sample_custom:
                self.__on_sample_custom(sample)
        except:
            pass
