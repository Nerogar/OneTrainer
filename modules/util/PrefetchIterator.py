import queue
import threading
from collections.abc import Iterable, Iterator
from contextlib import nullcontext, suppress

import torch


class PrefetchIterator:
    """Iterable wrapper that prefetches items ahead on a single background thread.

    Wrapping an iterable in PrefetchIterator lets the producer-side work
    (e.g. disk reads, decoding, encoding) overlap with whatever the consumer
    is doing between iterations.

    The producer runs on a dedicated CUDA stream so tensor uploads to the GPU
    don't have to wait for in-flight training work on the default stream.
    """

    def __init__(self, iterable: Iterable, queue_size: int = 1, stop_poll_interval: float = 0.1):
        self._iterable = iterable
        self._queue_size = queue_size
        # How often the producer checks the stop signal while blocked on put.
        self._stop_poll_interval = stop_poll_interval

    def __iter__(self) -> Iterator:
        q: queue.Queue = queue.Queue(maxsize=self._queue_size)
        stop_event = threading.Event()

        stream_ctx = torch.cuda.stream(torch.cuda.Stream()) if torch.cuda.is_available() else nullcontext()

        def put_or_stop(value) -> bool:
            # Block on put, but periodically wake to check the stop signal so
            # we can exit if the consumer has gone away.
            while not stop_event.is_set():
                with suppress(queue.Full):
                    q.put(value, timeout=self._stop_poll_interval)
                    return True
            return False

        def producer():
            with stream_ctx:
                try:
                    for item in self._iterable:
                        if not put_or_stop(item):
                            return
                except BaseException as e:
                    put_or_stop(e)
                    return
                put_or_stop(StopIteration())

        t = threading.Thread(target=producer, daemon=True)
        t.start()

        try:
            while True:
                item = q.get()
                if isinstance(item, StopIteration):
                    return
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            # Signal the producer to stop and drain anything pending so it
            # can wake from a blocked put and observe the stop signal.
            stop_event.set()
            with suppress(queue.Empty):
                while True:
                    q.get_nowait()
            t.join()
