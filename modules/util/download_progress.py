"""Forward Hugging Face Hub download progress to a callback.

Models given as a repository name are downloaded on first use. huggingface_hub only shows
that as a tqdm bar in the console, so the UI looks stuck on "loading the model" for minutes.
"""
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager

from huggingface_hub import file_download

# callback(file_name, downloaded_bytes, total_bytes or None)
DownloadCallback = Callable[[str, int, int | None], None]

_MIN_INTERVAL = 0.5  # seconds between callbacks for the same file

_lock = threading.Lock()
_callbacks: list[DownloadCallback] = []
_original_progress_bar_context = file_download._get_progress_bar_context


def _notify(name: str, done: int, total: int | None):
    with _lock:
        callbacks = list(_callbacks)
    for callback in callbacks:
        callback(name, done, total)


def _progress_bar_context(**kwargs):
    context = _original_progress_bar_context(**kwargs)
    if kwargs.get("_tqdm_bar") is not None or kwargs.get("unit", "B") != "B":
        return context  # a reused bar was already wrapped when it was created

    @contextmanager
    def wrapped():
        with context as bar:
            name = kwargs.get("desc") or ""
            total = kwargs.get("total")
            original_update = bar.update
            last_report = 0.0
            # counted here because a disabled bar (no console attached) never advances bar.n
            done = kwargs.get("initial") or 0

            def update(n=1):
                nonlocal last_report, done
                result = original_update(n)
                done += n or 0
                now = time.monotonic()
                if now - last_report >= _MIN_INTERVAL or (total and done >= total):
                    last_report = now
                    _notify(name, done, total)
                return result

            bar.update = update
            yield bar

    return wrapped()


@contextmanager
def report_downloads(callback: DownloadCallback) -> Iterator[None]:
    """Call `callback` with download progress for every Hub download started inside this block."""
    with _lock:
        _callbacks.append(callback)
        file_download._get_progress_bar_context = _progress_bar_context
    try:
        yield
    finally:
        with _lock:
            _callbacks.remove(callback)
            if not _callbacks:
                file_download._get_progress_bar_context = _original_progress_bar_context
