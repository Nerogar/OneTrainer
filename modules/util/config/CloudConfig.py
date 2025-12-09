from pathlib import Path
from typing import Any

from modules.util.config.BaseConfig import BaseConfig
from modules.util.enum.CloudAction import CloudAction
from modules.util.enum.CloudFileSync import CloudFileSync
from modules.util.enum.CloudType import CloudType


class CloudSecretsConfig(BaseConfig):
    api_key: str
    host: str
    port: int
    user: str
    id: str
    key_file: str
    password: str

    def __init__(self, data: list[(str, Any, type, bool)]):
        super().__init__(data)

    @staticmethod
    def default_values():
        data = []

        data.append(("api_key", "", str, False))
        data.append(("id", "", str, False))
        data.append(("host", "", str, False))
        data.append(("port", 0, str, False))
        data.append(("user", "root", str, False))
        data.append(("key_file", "", str, False)) # whilst not a secret, makes more semantic sense here
        data.append(("password", "", str, False))
        return CloudSecretsConfig(data)

    def expanded_key_file(self) -> str:
        key_file = getattr(self, "key_file", "").strip()
        if key_file == "":
            return ""
        return str(Path(key_file).expanduser())

    def connect_kwargs(self) -> dict[str, str]:
        kwargs: dict[str, str] = {}
        key_file = self.expanded_key_file()
        if key_file:
            kwargs["key_filename"] = key_file
        password = getattr(self, "password", "").strip()
        if password:
            kwargs["password"] = password
        return kwargs


class CloudConfig(BaseConfig):
    enabled: bool
    type: CloudType
    file_sync : CloudFileSync
    create : bool
    name: str
    tensorboard_tunnel: bool
    sub_type: str
    gpu_type: str
    volume_size: int
    min_download: int
    remote_dir: str
    huggingface_cache_dir: str
    onetrainer_dir: str
    install_onetrainer: bool
    update_onetrainer: bool
    install_cmd: str
    detach_trainer: bool
    run_id: str
    download_sampels : bool
    download_output_model : bool
    download_saves : bool
    download_backups : bool
    download_tensorboard : bool
    on_finish: CloudAction
    on_error: CloudAction
    on_detached_finish: CloudAction
    on_detached_error: CloudAction


    def __init__(self, data: list[(str, Any, type, bool)]):
        super().__init__(data)

    @staticmethod
    def default_values():
        data = []

        data.append(("enabled", False, bool, False))
        data.append(("type", CloudType.RUNPOD, CloudType, False))
        data.append(("file_sync", CloudFileSync.NATIVE_SCP, CloudFileSync, False))
        data.append(("create", True, bool, False))
        data.append(("name", "OneTrainer", str, False))
        data.append(("tensorboard_tunnel", True, bool, False))
        data.append(("sub_type", "", str, False))
        data.append(("gpu_type", "", str, False))
        data.append(("volume_size", 100, int, False))
        data.append(("min_download", 0, int, False))
        data.append(("remote_dir", "/workspace", str, False))
        data.append(("huggingface_cache_dir", "/workspace/huggingface_cache", str, False))
        data.append(("onetrainer_dir", "/workspace/OneTrainer", str, False))
        data.append(("install_cmd", "git clone https://github.com/Nerogar/OneTrainer", str, False))
        data.append(("install_onetrainer", True, bool, False))
        data.append(("update_onetrainer", True, bool, False))
        data.append(("detach_trainer", False, bool, False))
        data.append(("run_id", "job1", str, False))
        data.append(("download_samples", True, bool, False))
        data.append(("download_output_model", True, bool, False))
        data.append(("download_saves", True, bool, False))
        data.append(("download_backups", False, bool, False))
        data.append(("download_tensorboard", False, bool, False))
        data.append(("delete_workspace", False, bool, False))
        data.append(("on_finish", CloudAction.NONE, CloudAction, False))
        data.append(("on_error", CloudAction.NONE, CloudAction, False))
        data.append(("on_detached_finish", CloudAction.NONE, CloudAction, False))
        data.append(("on_detached_error", CloudAction.NONE, CloudAction, False))
        return CloudConfig(data)
