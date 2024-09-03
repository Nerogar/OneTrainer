import platform

import torch


class TorchMemoryRecorder:

    def __init__(self, filename: str = "memory.pickle", enabled: bool = True):
        self.filename = filename
        self.enabled = enabled and platform.system() == 'Linux'

    def __enter__(self):
        if self.enabled:
            torch.cuda.memory._record_memory_history()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enabled:
            try:
                torch.cuda.memory._dump_snapshot(filename=self.filename)
                print(f"dumped memory snapshot to {self.filename}")
            except Exception:
                print(f"could not dump memory snapshot {self.filename}")

            torch.cuda.memory._record_memory_history(enabled=None)
