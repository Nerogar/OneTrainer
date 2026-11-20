from collections.abc import Callable
from dataclasses import dataclass


def get_multi_gpu_availability(local_gpu_count: int | None, cloud_enabled: bool) -> tuple[bool, str]:
    if cloud_enabled or local_gpu_count is not None and local_gpu_count >= 2:
        return True, "Enable multi-GPU training"
    if local_gpu_count is None:
        return False, "Multi-GPU is unavailable because GPU detection failed."
    return False, f"Multi-GPU requires at least two visible GPUs; {local_gpu_count} detected on this computer."


@dataclass(kw_only=True)
class GeneralTabController:
    local_gpu_count: int | None
    on_workspace_dir_change: Callable[[str], None]
    on_always_on_tensorboard_toggle: Callable[[], None]

    def get_multi_gpu_availability(self, cloud_enabled: bool) -> tuple[bool, str]:
        return get_multi_gpu_availability(self.local_gpu_count, cloud_enabled)
