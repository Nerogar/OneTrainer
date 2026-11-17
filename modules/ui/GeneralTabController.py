from collections.abc import Callable
from dataclasses import dataclass


@dataclass(kw_only=True)
class GeneralTabController:
    local_gpu_count: int | None
    on_workspace_dir_change: Callable[[str], None]
    on_always_on_tensorboard_toggle: Callable[[], None]

    def get_multi_gpu_availability(self, cloud_enabled: bool) -> tuple[bool, str]:
        count = self.local_gpu_count
        if cloud_enabled or count is not None and count >= 2:
            return True, "Enable multi-GPU training"
        if count is None:
            return False, "Multi-GPU is unavailable because GPU detection failed."
        return False, f"Multi-GPU requires at least two visible GPUs; {count} detected on this computer."
