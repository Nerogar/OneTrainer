import inspect
import sys
import threading
import traceback
import uuid

from modules.ui.models.StateModel import StateModel

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot


class BaseWorker(QObject):
    initialized = Signal()
    finished = Signal(str)
    errored = Signal(tuple)
    aborted = Signal()
    result = Signal(object)
    progress = Signal(dict) # Arbitrary key-value pairs.

    def __init__(self):
        super().__init__()


    def progressCallback(self):
        def f(data):
            self.progress.emit(data)
        return f

    def _threadWrapper(self):
        try:
            self.initialized.emit()
            if self.inject_progress_callback:
                if "progress_fn" not in inspect.signature(self.fn).parameters:
                    print("WARNING: callable function has no progress_fn parameter. Invoking the function without it.")
                    out = self.fn(**self.kwargs)
                else:
                    out = self.fn(progress_fn=self.progressCallback(), **self.kwargs)
            else:
                out = self.fn(**self.kwargs)

            if self.abort_flag is not None and self.abort_flag.is_set():
                self.abort_flag.clear()
                self.aborted.emit()
        except Exception:
            StateModel.instance().log("critical", traceback.format_exc())
            exctype, value = sys.exc_info()[:2]
            self.errored.emit((exctype, value, traceback.format_exc()))
        else:
            self.result.emit(out)
        finally:
            self.finished.emit(self.name)

    def connectCallbacks(self, init_fn=None, result_fn=None, finished_fn=None, errored_fn=None, aborted_fn=None, progress_fn=None):
        if init_fn is not None:
            self.connections["initialized"].append(self.initialized.connect(init_fn))
        if result_fn is not None:
            self.connections["result"].append(self.result.connect(result_fn))
        if errored_fn is not None:
            self.connections["errored"].append(self.errored.connect(errored_fn))
        if finished_fn is not None:
            self.connections["finished"].append(self.finished.connect(finished_fn))
        if aborted_fn is not None:
            self.connections["aborted"].append(self.aborted.connect(aborted_fn))
        if progress_fn is not None:
            self.connections["progress"].append(self.progress.connect(progress_fn))

    def disconnectAll(self):
        for v in self.connections.values():
            for v2 in v:
                v2.disconnect()
        self.connections = {"initialized": [], "result": [], "errored": [], "finished": [], "aborted": [], "progress": []}


# Thread Worker based on QRunnable (it cannot be joined, but it is automatically enqueued on QT6's QThreadPool, balancing loads automatically.
# IMPORTANT: For severe exceptions (e.g., CUDA errors) it may crash the entire application with SIGSEGV.
# According to this: https://stackoverflow.com/questions/59837773/qtcore-qrunnable-causes-sigsev-pyqt5
# The problem may be that multiple inheritance may cause sometimes to access reserved memory
class RunnableWorker(QRunnable, BaseWorker):
    def __init__(self, fn, name, abort_flag=None, inject_progress_callback=False, **kwargs):
        QRunnable.__init__(self)

        self.fn = fn
        self.name = name
        self.abort_flag = abort_flag
        self.kwargs = kwargs
        self.inject_progress_callback = inject_progress_callback

        self.connections = {"initialized": [], "result": [], "errored": [], "finished": [], "aborted": [], "progress": []}
        self.destroyed.connect(lambda _: self.disconnectAll)

    @Slot()
    def run(self):
        self._threadWrapper()

# Thread Worker based on threading.Thread, it is a manually managed thread, with join capabilities.
# It *should* survive severe exceptions, as it is a native python implementation.
class PoolLessWorker(BaseWorker):
    def __init__(self, fn, name, abort_flag=None, inject_progress_callback=False, daemon=False, **kwargs):
        BaseWorker.__init__(self)

        self.fn = fn
        self.name = name
        self.abort_flag = abort_flag
        self.kwargs = kwargs
        self.inject_progress_callback = inject_progress_callback

        self.connections = {"initialized": [], "result": [], "errored": [], "finished": [], "aborted": [], "progress": []}
        self.destroyed.connect(lambda _: self.disconnectAll)

        self._thread = threading.Thread(target=self._threadWrapper, daemon=daemon)

    def start(self):
        self._thread.start()

    def join(self, timeout=None):
        self._thread.join(timeout)

    def isAlive(self):
        return self._thread.is_alive()



# Simple worker pool class. It allows to enqueue arbitrary functions executed on a QThreadPool. All the function parameters must be passed BY NAME (kwargs).
# If a job is associated with a name (createNamed()), its execution is reentrant (i.e., attempting to run the same job multiple times, will execute it only once).
# Workers (returned by createNamed and createAnonymous) expose initialized(), finished(), aborted(), result(function output) and errored(exception, value, traceback) signals.
# Abort events are a responsibility of the function, which can optionally be associated with a threading.Event() object (the aborted signal will be emitted if at the end of the execution, the event is_set()).
# IMPORTANT: the finished signal also removes the worker reference from this class, therefore unless a reference is saved somewhere else, it will be garbage collected.
# Using the worker's connect() method should avoid errors due to connections still active after garbage collection.
#
# A typical worker life-cycle is:
# worker_object, worker_id = WorkerPool.instance().createNamed(...)/createAnonymous(...)
# worker_object.connect(...)
# WorkerPool.instance().start(worker_id)

class WorkerPool:
    _instance = None

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.pool = QThreadPool()
        self.named_workers = {} # This worker's list refuses to append a new worker with the same name.
        self.anonymous_workers = {} # This worker's list can grow arbitrarily.
        self.poolless_workers = {}

    def __len__(self):
        return len(self.anonymous_workers) + len(self.named_workers)

    def createAnonymous(self, fn, abort_flag=None, **kwargs):
        id = str(uuid.uuid4())
        worker = RunnableWorker(fn, id, abort_flag=abort_flag, **kwargs)
        worker.connectCallbacks(finished_fn=self.__removeFinished(is_named=False))
        self.anonymous_workers[id] = worker
        return worker, id

    def createNamed(self, fn, name, poolless=False, daemon=False, abort_flag=None, **kwargs):
        if name not in self.named_workers:
            if poolless:
                worker = PoolLessWorker(fn, name, abort_flag=abort_flag, daemon=daemon, **kwargs)
                worker.connectCallbacks(finished_fn=self.__removeFinished(is_named=True))
                self.poolless_workers[name] = worker
            else:
                worker = RunnableWorker(fn, name, abort_flag=abort_flag, **kwargs)
                worker.connectCallbacks(finished_fn=self.__removeFinished(is_named=True))
                self.named_workers[name] = worker
            return worker, name
        else:
            return None, None


    def start(self, worker_id):
        ok = False
        if worker_id in self.named_workers:
            ok = True
            self.pool.start(self.named_workers[worker_id])
        elif worker_id in self.poolless_workers:
            ok = True
            self.poolless_workers[worker_id].start()
        elif worker_id in self.anonymous_workers:
            ok = True
            self.pool.start(self.anonymous_workers[worker_id])

        return ok

    def __removeFinished(self, is_named):
        def f(name):
            if is_named:
                if name in self.named_workers:
                    self.named_workers.pop(name)
                elif name in self.poolless_workers:
                    self.poolless_workers.pop(name)
            else:
                self.anonymous_workers.pop(name)
        return f
