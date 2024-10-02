import contextlib
from collections.abc import Callable

from modules.util.TrainProgress import TrainProgress

from PIL.Image import Image


class TrainCallbacks:
    def __init__(
            self,
            on_update_train_progress: Callable[[TrainProgress, int, int], None] = lambda _, __, ___: None,
            on_update_status: Callable[[str], None] = lambda _: None,
            on_sample_default: Callable[[Image], None] = lambda _: None,
            on_update_sample_default_progress: Callable[[int, int], None] = lambda _, __: None,
            on_sample_custom: Callable[[Image], None] = lambda _: None,
            on_update_sample_custom_progress: Callable[[int, int], None] = lambda _, __: None,
    ):
        self.__on_update_train_progress = on_update_train_progress
        self.__on_update_status = on_update_status
        self.__on_sample_default = on_sample_default
        self.__on_update_sample_default_progress = on_update_sample_default_progress
        self.__on_sample_custom = on_sample_custom
        self.__on_update_sample_custom_progress = on_update_sample_custom_progress

    # on_update_train_progress
    def set_on_update_train_progress(
            self,
            on_update_train_progress: Callable[[TrainProgress, int, int], None] = lambda _, __, ___: None,
    ):
        self.__on_update_train_progress = on_update_train_progress

    def on_update_train_progress(self, train_progress: TrainProgress, max_sample: int, max_epoch: int):
        if self.__on_update_train_progress:
            with contextlib.suppress(Exception):
                self.__on_update_train_progress(train_progress, max_sample, max_epoch)

    # on_update_status
    def set_on_update_status(
            self,
            on_update_status: Callable[[str], None] = lambda _: None,
    ):
        self.__on_update_status = on_update_status

    def on_update_status(self, status: str):
        if self.__on_update_status:
            with contextlib.suppress(Exception):
                self.__on_update_status(status)

    # on_sample_default
    def set_on_sample_default(
            self,
            on_sample_default: Callable[[Image], None] = lambda _: None,
    ):
        self.__on_sample_default = on_sample_default

    def on_sample_default(self, sample: Image):
        if self.__on_sample_default:
            with contextlib.suppress(Exception):
                self.__on_sample_default(sample)

    # on_update_sample_default_progress
    def set_on_update_sample_default_progress(
            self,
            on_update_sample_default_progress: Callable[[int, int], None] = lambda _, __: None,
    ):
        self.__on_update_sample_default_progress = on_update_sample_default_progress

    def on_update_sample_default_progress(self, step: int, max_step: int):
        if self.__on_update_sample_default_progress:
            with contextlib.suppress(Exception):
                self.__on_update_sample_default_progress(step, max_step)

    # on_sample_custom
    def set_on_sample_custom(
            self,
            on_sample_custom: Callable[[Image], None] = lambda _: None,
    ):
        self.__on_sample_custom = on_sample_custom

    def on_sample_custom(self, sample: Image):
        if self.__on_sample_custom:
            with contextlib.suppress(Exception):
                self.__on_sample_custom(sample)

    # on_update_sample_custom_progress
    def set_on_update_sample_custom_progress(
            self,
            on_update_sample_custom_progress: Callable[[int, int], None] = lambda _, __: None,
    ):
        self.__on_update_sample_custom_progress = on_update_sample_custom_progress

    def on_update_sample_custom_progress(self, progress: int, max_progress: int):
        if self.__on_update_sample_custom_progress:
            with contextlib.suppress(Exception):
                self.__on_update_sample_custom_progress(progress, max_progress)
