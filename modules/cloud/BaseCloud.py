import json
from abc import ABCMeta, abstractmethod
from pathlib import Path

from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.config.CloudConfig import CloudConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.time_util import get_string_timestamp


class BaseCloud(metaclass=ABCMeta):
    def __init__(self, config: TrainConfig):
        super().__init__()
        self.config = config


    def setup(self):
        self._connect()

        if self.config.cloud.install_onetrainer:
            self._install_onetrainer()

        if self.config.cloud.tensorboard_tunnel:
            self._make_tensorboard_tunnel()

    def download_output_model(self):
        local=Path(self.config.local_output_model_destination)
        remote=Path(self.config.output_model_destination)
        self._download_file(local=local,remote=remote)
        self._download_dir(local=local.with_suffix(local.suffix+"_embeddings"),
                           remote=remote.with_suffix(remote.suffix+"_embeddings"))

    def upload_config(self,commands : TrainCommands=None):
        local_config_path=Path(self.config.local_workspace_dir,f"remote_config-{get_string_timestamp()}.json")
        with local_config_path.open(mode="w") as f:
            json.dump(self.config.to_pack_dict(), f, indent=4)
        self._upload_config_file(local_config_path)

        if hasattr(self.config,"local_base_model_name"):
            self._upload(local=Path(self.config.local_base_model_name),remote=Path(self.config.base_model_name))
        if self.config.lora_model_name != "":
            self._upload(local=Path(self.config.local_lora_model_name),remote=Path(self.config.lora_model_name))

        if self.config.embedding.model_name != "":
            self._upload(local=Path(self.config.embedding.local_model_name),remote=Path(self.config.embedding.model_name))
        for add_embedding in self.config.additional_embeddings:
            if add_embedding.model_name != "":
                self._upload(local=Path(add_embedding.local_model_name),remote=Path(add_embedding.model_name))

        for concept in self.config.concepts:
            print(f"uploading concept {concept.name}...")
            if commands and commands.get_stop_command():
                return
            self._upload(local=Path(concept.local_path),remote=Path(concept.path),commands=commands)
            if len(concept.text.local_prompt_path) > 0:
                self._upload(local=Path(concept.text.local_prompt_path),remote=Path(concept.text.prompt_path))

    @staticmethod
    def _filter_download(config : CloudConfig,path : Path):
        if 'samples' in path.parts:
            return config.download_samples
        elif 'save' in path.parts:
            return config.download_saves
        elif 'backup' in path.parts:
            return config.download_backups
        elif 'tensorboard' in path.parts:
            return config.download_tensorboard
        else:
            return True


    @abstractmethod
    def run_trainer(self):
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def exec_callback(self,callbacks : TrainCallbacks):
        pass

    @abstractmethod
    def send_commands(self,commands : TrainCommands):
        pass

    @abstractmethod
    def sync_workspace(self):
        pass

    @abstractmethod
    def can_reattach(self):
        pass

    def _create(self):
        raise NotImplementedError("creating clouds not supported for this cloud type")

    def delete(self):
        raise NotImplementedError("deleting this cloud type not supported")

    def stop(self):
        raise NotImplementedError("stopping this cloud type not supported")

    @abstractmethod
    def _install_onetrainer(self):
        pass

    @abstractmethod
    def _make_tensorboard_tunnel(self):
        raise NotImplementedError("Tensorboard tunnel not supported on this cloud type")

    @abstractmethod
    def _upload(self,local : Path,remote : Path,commands : TrainCommands = None):
        #only call with main thread!
        pass

    @abstractmethod
    def _upload_config_file(self,local : Path):
        pass

    @abstractmethod
    def _download_file(self,local : Path,remote : Path):
        #only call with main thread!
        pass

    @abstractmethod
    def _download_dir(self,local : Path,remote : Path):
        #only call with main thread!
        pass

    @abstractmethod
    def delete_workspace(self):
        pass
