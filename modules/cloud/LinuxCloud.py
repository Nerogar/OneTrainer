import pickle
import shlex
import threading
import time
from pathlib import Path

from modules.cloud.BaseCloud import BaseCloud
from modules.cloud.FabricFileSync import FabricFileSync
from modules.cloud.NativeSCPFileSync import NativeSCPFileSync
from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.CloudAction import CloudAction
from modules.util.enum.CloudFileSync import CloudFileSync
from modules.util.time_util import get_string_timestamp

import fabric


class LinuxCloud(BaseCloud):
    def __init__(self, config: TrainConfig):
        super().__init__(config)
        self.connection=None
        self.callback_connection=None
        self.command_connection=None
        self.tensorboard_tunnel_stop=None

        name=config.cloud.run_id if config.cloud.detach_trainer else get_string_timestamp()
        self.callback_file=f'{config.cloud.remote_dir}/{name}.callback'
        self.command_pipe=f'{config.cloud.remote_dir}/{name}.command'
        self.config_file=f'{config.cloud.remote_dir}/{name}.json'
        self.exit_status_file=f'{config.cloud.remote_dir}/{name}.exit'
        self.log_file=f'{config.cloud.remote_dir}/{name}.log'
        self.pid_file=f'{config.cloud.remote_dir}/{name}.pid'

    def _connect(self):
        if self.connection:
            return

        config=self.config.cloud
        secrets=self.config.secrets.cloud
        connect_kwargs=secrets.connect_kwargs()

        if secrets.host == '' or secrets.port == '':
            raise ValueError('Host and port required for SSH connection')

        try:
            self.connection=fabric.Connection(host=secrets.host,
                                             port=secrets.port,
                                             user=secrets.user,
                                             connect_kwargs=connect_kwargs)
            self.connection.open()
            self.connection.transport.set_keepalive(30)

            self.callback_connection=fabric.Connection(host=secrets.host,
                                                       port=secrets.port,
                                                       user=secrets.user,
                                                       connect_kwargs=connect_kwargs)

            self.command_connection=fabric.Connection(host=secrets.host,
                                                      port=secrets.port,
                                                      user=secrets.user,
                                                      connect_kwargs=connect_kwargs)
            #the command connection isn't used for long periods of time; prevent remote from closing it:
            self.command_connection.open()
            self.command_connection.transport.set_keepalive(30)

            match config.file_sync:
                case CloudFileSync.NATIVE_SCP:
                    self.file_sync=NativeSCPFileSync(config,secrets)
                case CloudFileSync.FABRIC_SFTP:
                    self.file_sync=FabricFileSync(config,secrets)

        except Exception:
            if self.connection:
                self.connection.close()
                self.connection=None
            if self.command_connection:
                self.command_connection.close()
            raise


    def setup(self):
        super().setup()
        self.connection.run(f'mkfifo {shlex.quote(self.command_pipe)}',warn=True,hide=True,in_stream=False)

    def _install_onetrainer(self, update: bool=False):
        config=self.config.cloud
        parent=Path(config.onetrainer_dir).parent.as_posix()
        self.connection.run(f'test -e {shlex.quote(config.onetrainer_dir)} \
                              || (mkdir -p {shlex.quote(parent)} \
                                  && cd {shlex.quote(parent)} \
                                  && {config.install_cmd})',in_stream=False)

        result=self.connection.run(f"test -d {shlex.quote(config.onetrainer_dir)}/venv",warn=True,in_stream=False)

        #many docker images, including the default ones on RunPod and vast.ai, only set up $PATH correctly
        #for interactive shells. On RunPod, cuda is missing from $PATH; on vast.ai, python is missing.
        #We cannot pretend to be interactive either, because then vast.ai starts a tmux screen.
        #Add these paths manually:
        cmd_env = f"export PATH=$PATH:/usr/local/cuda/bin:/venv/main/bin \
                   && export OT_LAZY_UPDATES=true \
                   && cd {shlex.quote(config.onetrainer_dir)}"

        if result.exited == 0:
            if update:
                self.connection.run(cmd_env + "&& ./update.sh", in_stream=False)
        else:
            self.connection.run(cmd_env + "&& ./install.sh", in_stream=False)

    def _make_tensorboard_tunnel(self):
        self.tensorboard_tunnel_stop=threading.Event()
        self.tensorboard_tunnel=fabric.tunnels.TunnelManager(
            local_host='localhost',
            local_port=self.config.tensorboard_port,
            remote_host='localhost',
            remote_port=self.config.tensorboard_port,
            transport=self.connection.client.get_transport(),
            finished=self.tensorboard_tunnel_stop
        )
        self.tensorboard_tunnel.start()


    def close(self):
        if self.tensorboard_tunnel_stop is not None:
            self.tensorboard_tunnel_stop.set()
        if self.callback_connection:
            self.callback_connection.close()
        if self.command_connection:
            self.command_connection.close()
        if self.file_sync:
            self.file_sync.close()
        if self.connection:
            self.connection.close()
            self.connection=None

    def can_reattach(self):
        result=self.connection.run(f"test -f {self.pid_file}",warn=True,in_stream=False)
        return result.exited == 0

    def _get_action_cmd(self,action : CloudAction):
        if action != CloudAction.NONE:
            raise NotImplementedError("Action on detached not supported for this cloud type")
        return ":"

    def run_trainer(self):
        config=self.config.cloud
        if self.can_reattach():
            self.__trail_detached_trainer()
            return

        cmd="export PATH=$PATH:/usr/local/cuda/bin:/venv/main/bin \
             && export PYTHONUNBUFFERED=1 \
             && export OT_LAZY_UPDATES=true"

        if self.config.secrets.huggingface_token != "":
            cmd+=f" && export HF_TOKEN={self.config.secrets.huggingface_token}"
        if config.huggingface_cache_dir != "":
            cmd+=f" && export HF_HOME={config.huggingface_cache_dir}"

        cmd+=f' && {config.onetrainer_dir}/run-cmd.sh train_remote --config-path={shlex.quote(self.config_file)} \
                                                                   --callback-path={shlex.quote(self.callback_file)} \
                                                                   --command-path={shlex.quote(self.command_pipe)}'

        if config.detach_trainer:
            self.connection.run(f'rm -f {self.exit_status_file}',in_stream=False)

            cmd=f"({cmd} ; exit_status=$? ; echo $exit_status > {self.exit_status_file}; exit $exit_status)"

            #if the callback file still exists 10 seconds after the trainer has exited, the client must be detached, because the clients reads and deletes this file:
            cmd+=f" && (sleep 10 && test -f {shlex.quote(self.callback_file)} && {self._get_action_cmd(config.on_detached_finish)} || true) \
                    || (sleep 10 && test -f {shlex.quote(self.callback_file)} && {self._get_action_cmd(config.on_detached_error)})"

            cmd=f'(nohup true && {cmd}) > {self.log_file} 2>&1 & echo $! > {self.pid_file}'
            self.connection.run(cmd,disown=True)
            self.__trail_detached_trainer()
        else:
            self.connection.run(cmd,in_stream=False)

    def __trail_detached_trainer(self):
        cmd=f'tail -f {self.log_file} --pid $(<{self.pid_file})'
        self.connection.run(cmd,in_stream=False)
        #trainer has exited, don't reattach:
        self.connection.run(f'rm -f {self.pid_file}',in_stream=False)
        #raise an exception if the training process return an exit code != 0:
        self.connection.run(f'exit $(<{self.exit_status_file})',in_stream=False)




    def exec_callback(self,callbacks : TrainCallbacks):
        #callbacks are a file instead of a named pipe, because of the blocking behaviour of linux pipes:
        #writing to pipes on the cloud can slow down training, and would cause issues in case
        #of a detached cloud trainer.
        #
        #use 'rename' as the atomic operation, to avoid race conditions between reader and writer:

        file=f'{shlex.quote(self.callback_file)}'
        cmd=f'mv "{file}" "{file}.read" \
           && cat "{file}.read" \
           && rm "{file}.read"'

        self.callback_connection.open()
        in_file,out_file,err_file=self.callback_connection.client.exec_command(cmd)

        try:
            while True:
                try:
                    while not out_file.channel.recv_ready() and not out_file.channel.exit_status_ready():
                        #even though reading from out_file is blocking, it doesn't block if there
                        #is *no* data available yet, which results in an unpickling error.
                        #wait until there is at least some data before reading:
                        time.sleep(0.1)
                    name=pickle.load(out_file)
                except EOFError:
                    return
                params=pickle.load(out_file)

                fun=getattr(callbacks,name)
                fun(*params)
        finally:
            in_file.close()
            out_file.close()
            err_file.close()


    def send_commands(self,commands : TrainCommands):
        try:
            self.command_connection.open()
            in_file,out_file,err_file=self.command_connection.client.exec_command(
                f'test -e {shlex.quote(self.command_pipe)} \
                && cat > {shlex.quote(self.command_pipe)}'
            )
            try:
                pickle.dump(commands,in_file)
                in_file.flush()
                in_file.channel.shutdown_write()
            finally:
                in_file.close()
                out_file.close()
                err_file.close()

            commands.reset()
        except Exception:
            if not self.command_connection.is_connected:
                print("\n\nCommand SSH connection lost. Attempting to reconnect...")
                self.command_connection.open()
                self.send_commands(commands)
            else:
                raise

    def _upload_config_file(self,local : Path):
        self.file_sync.sync_up_file(local,Path(self.config_file))

    def sync_workspace(self):
        self.file_sync.sync_down_dir(local=Path(self.config.local_workspace_dir),
                                  remote=Path(self.config.workspace_dir),
                                  filter=lambda path:BaseCloud._filter_download(config=self.config.cloud,path=path))

    def delete_workspace(self):
        self.connection.run(f"rm -r {shlex.quote(self.config.workspace_dir)}",in_stream=False)
