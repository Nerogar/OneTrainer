from collections.abc import Callable


class GeneralTabController:

    def __init__(
            self,
            *,
            local_gpu_count: int | None,
            on_workspace_dir_change: Callable[[str], None],
            on_always_on_tensorboard_toggle: Callable[[], None],
    ):
        self._local_gpu_count = local_gpu_count
        self._on_workspace_dir_change = on_workspace_dir_change
        self._on_always_on_tensorboard_toggle = on_always_on_tensorboard_toggle

    def get_multi_gpu_availability(self, cloud_enabled: bool) -> tuple[bool, str]:
        count = self._local_gpu_count
        if cloud_enabled or count is not None and count >= 2:
            return True, "Enable multi-GPU training"
        if count is None:
            return False, "Multi-GPU is unavailable because GPU detection failed."
        return False, f"Multi-GPU requires at least two visible GPUs; {count} detected on this computer."

    def on_workspace_dir_change(self, new_workspace_dir: str) -> None:
        self._on_workspace_dir_change(new_workspace_dir)

    def on_always_on_tensorboard_toggle(self) -> None:
        self._on_always_on_tensorboard_toggle()
