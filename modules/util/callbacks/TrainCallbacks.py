from typing import Callable

from PIL.Image import Image

from modules.util.TrainProgress import TrainProgress


class TrainCallbacks:
    def __init__(
            self,
            on_update_progress: Callable[[TrainProgress], None] = lambda _: None,
            on_sample: Callable[[Image], None] = lambda _: None,
    ):
        self.on_update_progress = on_update_progress
        self.on_sample = on_sample
