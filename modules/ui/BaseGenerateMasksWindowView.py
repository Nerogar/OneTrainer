from abc import ABC, abstractmethod


class BaseGenerateMasksWindowView(ABC):
    @abstractmethod
    def set_progress(self, value, max_value):
        pass
