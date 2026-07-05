import threading


class SingletonMixin:
    _instance: "SingletonMixin | None" = None
    _singleton_lock: threading.Lock = threading.Lock()

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        cls._instance = None
        cls._singleton_lock = threading.Lock()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._singleton_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
