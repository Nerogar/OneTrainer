import os #TODO only tensorboard
import subprocess #TODO only tensorboard
import sys #TODO only tensorboard
import threading
import time
import copy
import json

from pathlib import Path
from modules.trainer.BaseTrainer import BaseTrainer
from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.config.TrainConfig import TrainConfig
from modules.util.TrainProgress import TrainProgress
from modules.util.enum.CloudType import CloudType
from modules.util.enum.CloudAction import CloudAction
from modules.cloud.LinuxCloud import LinuxCloud
from modules.cloud.RunpodCloud import RunpodCloud
from modules.util.config.ConceptConfig import ConceptConfig


class CloudTrainer(BaseTrainer):
    tensorboard_subprocess: subprocess.Popen

    def __init__(self, config: TrainConfig, callbacks: TrainCallbacks, commands: TrainCommands):
        super(CloudTrainer, self).__init__(config, callbacks, commands)
        self.error_caught=False
        self.callback_thread=None
        self.sync_thread=None
        self.stop_event=None
        self.cloud=None
        self.remote_config=CloudTrainer.__make_remote_config(config)

        tensorboard_log_dir = os.path.join(config.workspace_dir, "tensorboard")
        os.makedirs(Path(tensorboard_log_dir).absolute(), exist_ok=True)
        if config.tensorboard:
            if config.cloud.tensorboard_tunnel:
                pass #TODO
            else:
                tensorboard_executable = os.path.join(os.path.dirname(sys.executable), "tensorboard")

                tensorboard_args = [
                    tensorboard_executable,
                    "--logdir",
                    tensorboard_log_dir,
                    "--port",
                    "6006",
                    "--samples_per_plugin=images=100,scalars=10000",
                ]

                if self.config.tensorboard_expose:
                    tensorboard_args.append("--bind_all")

                self.tensorboard_subprocess = subprocess.Popen(tensorboard_args)

        match config.cloud.type:
            case CloudType.RUNPOD: self.cloud=RunpodCloud(self.remote_config)
            case CloudType.LINUX: self.cloud=LinuxCloud(self.remote_config)

    def start(self):
        try:
            self.callbacks.on_update_status("setting up cloud")
            self.cloud.setup()

            if not self.cloud.can_reattach():
                self.callbacks.on_update_status("uploading config")
                self.cloud.upload_config(self.commands)
        except:
            self.error_caught=True
            raise
        
        def on_command(commands : TrainCommands):
            backup_on_command = commands.get_and_reset_on_command() #don't pickle a Callable
            self.cloud.send_commands(commands)
            commands.set_on_command(backup_on_command)
        self.commands.set_on_command(on_command)

        self.stop_event=threading.Event()

        def callback():
            while not self.stop_event.is_set():
                self.cloud.exec_callback(self.callbacks)
                time.sleep(1)
        self.callback_thread = threading.Thread(target=callback)
        self.callback_thread.start()
        
        def sync():
            while not self.stop_event.is_set():
                self.cloud.sync_workspace()
                time.sleep(5)
            self.cloud.sync_workspace()
        self.sync_thread = threading.Thread(target=sync)
        self.sync_thread.start()
                
        if self.config.continue_last_backup:
            print('warning: backups are not uploaded, but expected to be on the cloud already!')

    def train(self):
        if self.commands.get_stop_command(): return
        try:
            self.callbacks.on_update_status("starting trainer on cloud")
            self.cloud.run_trainer()

            if self.config.cloud.download_output_model:
                self.callbacks.on_update_status("downloading output model")
                self.cloud.download_output_model()
        except Exception:
            self.error_caught=True
            raise
        finally:
            self.stop_event.set()
            self.callback_thread.join()
            self.sync_thread.join()

    def end(self):
        try:
            if self.config.tensorboard and not self.config.cloud.tensorboard_tunnel: self.tensorboard_subprocess.kill()
            self.cloud.close()
        except Exception:
            self.error_caught=True
            raise
        finally:
            if self.error_caught: action=self.config.cloud.on_error
            elif self.commands.get_stop_command(): action=CloudAction.NONE
            else: action=self.config.cloud.on_finish
            
            if action == CloudAction.DELETE: self.cloud.delete()
            elif action == CloudAction.STOP: self.cloud.stop()
            
            del self.cloud

    @staticmethod
    def __make_remote_config(local : TrainConfig):
        CloudTrainer.__load_concepts(local)
        remote=copy.deepcopy(local)
        remote.cloud = local.cloud #share cloud config, so UI can be updated to IP, port, cloudid etc.
        
        def adjust(config,attribute : str):
            path=getattr(config,attribute)
            setattr(config,"local_"+attribute,path)
            path=CloudTrainer.__adjust_path(path,remote.cloud.workspace_dir)
            setattr(config,attribute,path)

        adjust(remote,"debug_dir")
        adjust(remote,"workspace_dir")
        adjust(remote,"cache_dir")
        if Path(remote.base_model_name).exists():
            adjust(remote,"base_model_name")

        adjust(remote,"output_model_destination")
        adjust(remote,"lora_model_name")

        #TODO embedding files
        
        
        remote.concept_file_name=""
        remote.concepts = [concept for concept in remote.concepts if concept.enabled]

        for concept in remote.concepts:
            adjust(concept,"path")
            adjust(concept.text,"prompt_path")

        return remote

    @staticmethod
    def __adjust_path(pathstr : str,remote_workspace_dir : str):
        if len(pathstr.strip()) > 0:
            path=Path(pathstr)
            if path.is_absolute(): path=path.relative_to(path.anchor)  #remove windows drive name C:
            return (Path(remote_workspace_dir,"remote") / path).as_posix()
        else: return ""

    @staticmethod
    def __load_concepts(config): #TODO is there not a function in OT that can do that?
        if config.concepts is None:
            with open(config.concept_file_name, 'r') as f:
                config.concepts=[]
                json_concepts = json.load(f)
                for json_concept in json_concepts:
                    concept=ConceptConfig.default_values().from_dict(json_concept)
                    config.concepts.append(concept)

    def backup(self, train_progress: TrainProgress):
        pass

    def save(self, train_progress: TrainProgress):
        pass

