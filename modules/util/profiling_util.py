import platform
import threading

import torch


def _private_ram_bytes() -> int:
    # Private, non-file-backed resident memory. Excludes mmap'd file pages -- notably the safetensors weights
    # that safe_open maps during load/materialize, which are reclaimable page cache and don't reflect real
    # memory pressure. This matches what system monitors report as "used" memory, unlike ru_maxrss / VmHWM
    # (Linux) or working-set (Windows), which count the mmap'd file cache too and so overstate usage on the
    # disk-offload path.
    if platform.system() == 'Linux':
        # RssAnon from /proc; reported in KiB.
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("RssAnon:"):
                    return int(line.split()[1]) * 1024
        raise RuntimeError("RssAnon not found in /proc/self/status")
    if platform.system() == 'Windows':
        # psutil's pmem.private is process-private bytes, i.e. Windows' analogue of anon RSS.
        import psutil
        return psutil.Process().memory_info().private
    raise RuntimeError(f"peak RAM tracking not supported on {platform.system()}")


class PeakMemoryRecorder:
    # The OS only tracks a peak for total RSS / working set, not for private/anon RSS, so peak RAM is found by
    # polling it on a daemon thread for the duration of the with-block. Peak VRAM needs no polling: the CUDA
    # caching allocator already tracks it, reset_peak_memory_stats() scopes it to this block.
    def __init__(self, label: str, enabled: bool = True, poll_interval: float = 0.1):
        self.label = label
        self.enabled = enabled
        self.ram_enabled = enabled and platform.system() in ('Linux', 'Windows')
        self.poll_interval = poll_interval
        self._peak_ram = 0
        self._stop_event = threading.Event()
        self._thread = None

    def __enter__(self):
        if self.ram_enabled:
            self._peak_ram = _private_ram_bytes()
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._poll_ram, daemon=True)
            self._thread.start()
        if self.enabled and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        return self

    def _poll_ram(self):
        while not self._stop_event.is_set():
            self._peak_ram = max(self._peak_ram, _private_ram_bytes())
            self._stop_event.wait(self.poll_interval)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.ram_enabled:
            self._stop_event.set()
            self._thread.join()
            # fold in one last sample in case the peak happened between the last poll and thread stop
            self._peak_ram = max(self._peak_ram, _private_ram_bytes())
            print(f"[peak_ram] {self.label}: {self._peak_ram / 1024 ** 3:.2f} GiB")

        if self.enabled and torch.cuda.is_available():
            allocated = torch.cuda.max_memory_allocated() / 1024 ** 3
            reserved = torch.cuda.max_memory_reserved() / 1024 ** 3
            print(f"[peak_vram] {self.label}: {allocated:.2f} GiB (reserved {reserved:.2f} GiB)")


class TorchMemoryRecorder:
    def __init__(self, filename: str = "memory.pickle", enabled: bool = True):
        self.filename = filename
        self.enabled = enabled and platform.system() == 'Linux'

    def __enter__(self):
        if self.enabled:
            # stacks="python" attributes each allocation to the python call site and is far cheaper than the
            # default C++ unwinding, which stalls badly under torch.compile. max_entries is a ring buffer: once
            # full, new alloc/free events overwrite the oldest ones rather than stopping collection, so this
            # keeps the most recent 100k events, bounding host RAM use on a long/heavy step.
            torch.cuda.memory._record_memory_history(context="all", stacks="python", max_entries=100_000)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enabled:
            try:
                torch.cuda.memory._dump_snapshot(filename=self.filename)
            except Exception:
                print(f"could not write memory snapshot {self.filename}")

            torch.cuda.memory._record_memory_history(enabled=None)

class TorchProfiler:
    def __init__(self, filename: str, enabled: bool = True):
        self.filename = filename
        self.enabled = enabled
        self.profiler = None

    def __enter__(self):
        if self.enabled:
            profiler_context = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA
                ],
            )
            self.profiler = profiler_context.__enter__()
            return self.profiler
        else:
            return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.profiler is not None:
            ret = self.profiler.__exit__(exc_type, exc_val, exc_tb)
            try:
                self.profiler.export_chrome_trace(self.filename)
            except Exception:
                print(f"could not write profiler output {self.filename}")
            return ret
        else:
            return False
