import platform

import torch


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
