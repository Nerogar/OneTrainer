from abc import ABC, abstractmethod


class BaseGenerateCaptionsWindowView(ABC):
    @abstractmethod
    def set_progress(self, value, max_value):
        pass
