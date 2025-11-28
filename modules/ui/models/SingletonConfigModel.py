import os
import threading
import traceback
from contextlib import contextmanager

from modules.util import path_util

# TODO: Cleanup
# 1) Naming convention is all over the place (camelCase, snake_case, inconsistent visibility __method vs _method vs method) -> Proposed convention: camelCase for Controller methods, snake_case for Model methods. Visibility: expose the minimum amount of methods.
# 2) Use self.log instead of print/logging.logger/traceback.print_exc()


# Base class for config models. It provides a Singleton interface and four synchronization mechanisms:
# - with self.critical_region_read(): allows free access to the "self" instance to any thread, as long as nobody is writing, otherwise waits for the writer to finish.
# - with self.critical_region-write(): waits for all the readers to finish, and then blocks the instance "self" for writing.
# - with self.critical_region(): blocks all the threads trying to access the "self" instance in a generic way.
# - with self.critical_region_global(): blocks ALL the model instances derived from this class.
# Ideally, only read/write accesses should be used, the other methods are there to cover only limited use cases, as they cause significant performance degradation.
# Logging, with the self.log() method, blocks all instances, albeit using a specific reentrant lock.
class SingletonConfigModel:
    _instance = None
    _frozenConfig = None
    _is_frozen = False

    # The following are reentrant locks shared across all the subclasses.
    _global_mutex = threading.RLock() # Generic.
    _log_mutex = threading.RLock() # Specific for logging messages.

    def __init__(self, config=None):
        self.config = config

        self._mutex = threading.RLock() # Local reentrant lock.

        # Reentrant read-write lock implementation based on: https://gist.github.com/icezyclon/124df594496dee71ce8455a31b1dd29f
        self._writer_id = None
        self._writer_count = 0
        self._readers = {}
        self._condition = threading.Condition(threading.RLock())

    def __acquire_read_lock(self):
        id = threading.get_ident()
        with self._condition:
            self._readers[id] = self._readers.get(id, 0) + 1

    def __release_read_lock(self):
        id = threading.get_ident()
        with self._condition:
            if id not in self._readers:
                raise RuntimeError(f"Read lock was released while not holding it by thread {id}")
            if self._readers[id] == 1:
                del self._readers[id]
            else:
                self._readers[id] -= 1

            if not self._readers:
                self._condition.notify()

    def __acquire_write_lock(self):
        id = threading.get_ident()

        self._condition.acquire()
        if self._writer_id == id:
            self._writer_count += 1
            return

        times_reading = self._readers.pop(id, 0)
        while len(self._readers) > 0:
            self._condition.wait()
        self._writer_id = id
        self._writer_count += 1
        if times_reading:
            self._readers[id] = times_reading

    def __release_write_lock(self):
        if self._writer_id != threading.get_ident():
            raise RuntimeError(f"Write lock was released while not holding it by thread {threading.current_thread().ident}")
        self._writer_count -= 1
        if self._writer_count == 0:
            self._writer_id = None
            self._condition.notify()
        self._condition.release()

    @contextmanager
    def critical_region_read(self):
        try:
            self.__acquire_read_lock()
            yield
        finally:
            self.__release_read_lock()

    @contextmanager
    def critical_region_write(self):
        try:
            self.__acquire_write_lock()
            yield
        finally:
            self.__release_write_lock()

    @contextmanager
    def critical_region_global(self):
        try:
            self._global_mutex.acquire()
            yield
        finally:
            self._global_mutex.release()

    @contextmanager
    def critical_region(self):
        try:
            self._mutex.acquire()
            yield
        finally:
            self._mutex.release()

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def log(self, severity, message):
        self._log_mutex.acquire()
        print(f"{severity}: {message}") # TODO: use logging.logger, logging on file, other approach?
        # Proposed severities: "critical", "error", "warning", "debug", "info"
        # Maybe some of them default to files, others to console, other both?
        self._log_mutex.release()

    # Read a list of config variables at once, in a thread-safe fashion.
    # Important: this method should be used at the beginning of long computations, to fetch a coherent collection of values, regardless of
    def bulk_read(self, *paths, as_dict=False):
        with self.critical_region_read():
            if as_dict:
                out = {path: self.get_state(path) for path in paths}
            else:
                out = [self.get_state(path) for path in paths]
        return out

    # Write a list of config variables at once, in a thread-safe fashion.
    def bulk_write(self, kv_pairs):
        with self.critical_region_write():
            for k, v in kv_pairs.items():
                self.set_state(k, v)


    # Read a single config variable in a thread-safe fashion.
    def get_state(self, path):
        if self.config is not None:
            try:
                with self.critical_region_read():
                    ref = self.config
                    if path == "":
                        return ref

                    for key in str(path).split("."):
                        if isinstance(ref, list):
                            ref = ref[int(key)]
                        elif isinstance(ref, dict) and key in ref:
                            ref = ref[key]
                        elif hasattr(ref, key):
                            ref = getattr(ref, key)
                        else:
                            self.log("debug", f"Key {key} not found in config")
                            return None
                    return ref

            except Exception:
                self.log("critical", traceback.format_exc())
        return None

    # Write a single config variable in a thread-safe fashion.
    def set_state(self, path, value):
        if self.config is not None:
            with self.critical_region_write():
                ref = self.config
                for ptr in str(path).split(".")[:-1]:
                    if isinstance(ref, list):
                        ref = ref[int(ptr)]
                    elif isinstance(ref, dict):
                        ref = ref[ptr]
                    elif hasattr(ref, ptr):
                        ref = getattr(ref, ptr)
                if isinstance(ref, list):
                    ref[int(path.split(".")[-1])] = value
                elif isinstance(ref, dict):
                    ref[path.split(".")[-1]] = value
                elif hasattr(ref, path.split(".")[-1]):
                    setattr(ref, path.split(".")[-1], value)
                else:
                    self.log("debug", f"Key {path} not found in config")

    def load_available_config_names(self, dir="training_presets", include_default=True):
        configs = [("", path_util.canonical_join(dir, "#.json"))] if include_default else []

        if os.path.isdir(dir):
            for path in os.listdir(dir):
                if path != "#.json":
                    path = path_util.canonical_join(dir, path)
                    if path.endswith(".json") and os.path.isfile(path):
                        name = os.path.basename(path)
                        name = os.path.splitext(name)[0]
                        configs.append((name, path))
            configs.sort()

        return configs
