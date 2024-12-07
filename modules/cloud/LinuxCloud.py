import pickle
import shlex
import socket
import threading
import time
from pathlib import Path

from modules.cloud.BaseCloud import BaseCloud
from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.config.CloudConfig import CloudConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.CloudAction import CloudAction
from modules.util.time_util import get_string_timestamp

import fabric


class LinuxCloud(BaseCloud):
    def __init__(self, config: TrainConfig):
        super().__init__(config)
        self.connection=None
        self.callback_connection=None
        self.command_connection=None
        self.sync_connection=None
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
        if config.host != '' and config.port != '':
            self.connection=fabric.Connection(host=config.host,port=config.port,user=config.user)
            self.connection.open()

            self.callback_connection=fabric.Connection(host=config.host,port=config.port,user=config.user)
            self.callback_connection.open()

            self.command_connection=fabric.Connection(host=config.host,port=config.port,user=config.user)
            self.command_connection.open()
            #the command connection isn't used for long periods of time; prevent remote from closing it:
            self.command_connection.transport.set_keepalive(30)

            self.sync_connection=fabric.Connection(host=config.host,port=config.port,user=config.user)
            self.sync_connection.open()
        else:
            raise ValueError('Host and port required for SSH connection')


    def setup(self):
        super().setup()
        self.connection.run(f'mkfifo {shlex.quote(self.command_pipe)}',warn=True,hide=True,in_stream=False)

    def _install_onetrainer(self):
        config=self.config.cloud
        parent=Path(config.onetrainer_dir).parent.as_posix()
        self.connection.run(f'test -d {shlex.quote(config.onetrainer_dir)} \
                              || (mkdir -p {shlex.quote(parent)} \
                                  && cd {shlex.quote(parent)} \
                                  && {config.install_cmd})',in_stream=False)

        #OT requires cuda in PATH, but runpod only sets that up in bashprofile, which is not used by fabric
        #TODO test with other clouds
        self.connection.run(f'test -d {shlex.quote(config.onetrainer_dir)}/venv \
                              || (cd {shlex.quote(config.onetrainer_dir)} \
                                  && export PATH=$PATH:/usr/local/cuda/bin \
                                  && ./install.sh)',in_stream=False)

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
        if self.sync_connection:
            self.sync_connection.close()
        if self.connection:
            self.connection.close()

    def can_reattach(self):
        result=self.connection.run(f"test -f {self.pid_file}",warn=True,in_stream=False)
        return result.exited == 0

    def _get_action_cmd(self,action : CloudAction):
        if action != CloudAction.NONE:
            raise NotImplementedError("Action on detached not supported for this cloud type")
        return ":"

    def run_trainer(self):
        config=self.config.cloud
        self.connection.put('scripts/train_remote.py',f'{config.onetrainer_dir}/scripts/train_remote.py') #TODO remove
        self.connection.put('modules/util/args/TrainArgs.py',f'{config.onetrainer_dir}/modules/util/args/TrainArgs.py') #TODO remove

        if self.can_reattach():
            print(f"Reattaching to run id {config.run_id}\n\n")
            self.__trail_detached_trainer()
            return


        cmd="export PYTHONUNBUFFERED=1"
        if config.huggingface_token != "":
            cmd+=f" && export HF_TOKEN={config.huggingface_token}"
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

        try:
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
        except Exception:
            if not self.callback_connection.is_connected:
                print("\n\nCallback SSH connection lost. Attempting to reconnect...")
                self.callback_connection.open()
                self.exec_callback(callbacks)
            else:
                raise


    def send_commands(self,commands : TrainCommands):
        try:
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


    def _upload(self,local : Path,remote : Path,commands : TrainCommands = None):
        update_info=LinuxCloud.__get_update_info(self.connection,remote)
        LinuxCloud.__upload(self.connection,local,remote,update_info,commands=commands)

    def _upload_config_file(self,local : Path):
        self._upload(local,Path(self.config_file))


    def _download_file(self,local : Path,remote : Path):
        update_info=LinuxCloud.__get_update_info(self.connection,remote)
        LinuxCloud.__download_file(self.connection,local,remote,update_info)

    def _download_dir(self,local : Path,remote : Path):
        update_info=LinuxCloud.__get_update_info(self.connection,remote)
        LinuxCloud.__download_dir(self.connection,local,remote,update_info)

    def sync_workspace(self):
        try:
            update_info=LinuxCloud.__get_update_info(self.sync_connection,Path(self.config.workspace_dir))
            LinuxCloud.__download_dir(self.sync_connection,
                                      local=Path(self.config.local_workspace_dir),
                                      remote=Path(self.config.workspace_dir),
                                      update_info=update_info,
                                      config=self.config.cloud)
        except Exception as e:
            if isinstance(e,socket.error):
                #Connection.is_connected is True even if SFTP socket is closed. Close and re-open:
                self.sync_connection.close()

            if not self.sync_connection.is_connected:
                print("\n\nSync SSH connection lost. Attempting to reconnect...")
                self.sync_connection.open()
                self.sync_workspace()
            else:
                raise

    @staticmethod
    def __get_update_info(connection,remote : Path):
        cmd = f'find {shlex.quote(remote.as_posix())} -exec stat --printf "%n\\t%s\\t%Y\\t%F\\n"' + ' {} \\;'
        result=connection.run(cmd,warn=True,hide=True,in_stream=False)
        # dict[filename]: (size,mtime,filetype)
        ret={}
        for line in result.stdout.splitlines():
            sp=line.split('\t')
            ret[Path(sp[0])]=(int(sp[1]),int(sp[2]),sp[3])
        return ret

    @staticmethod
    def __upload_file(connection,local : Path,remote : Path,update_info,commands=None):
        if remote in update_info:
            if (local.stat().st_size == update_info[remote][0]
                and local.stat().st_mtime <= update_info[remote][1]):
                    return

        if remote.parent not in update_info:
            connection.run(f'mkdir -p {shlex.quote(remote.parent.as_posix())}',in_stream=False)
            update_info[remote.parent]=(0,0,'directory')

        print(f'uploading {local}...')
        connection.put(str(local),remote.as_posix())

    @staticmethod
    def __upload_dir(connection,local : Path,remote : Path,update_info,commands : TrainCommands=None):
        n=sum(1 for entry in local.iterdir())
        if n > 50:
            print(f"WARNING: Uploading {n} files. Uploading many individual files can be slow via SSH.")
            print( "         If you prefer, you can interrupt the upload by pressing 'Stop Training',")
            print(f"         upload your files in {str(local)} manually to the cloud at {remote.as_posix()},")
            print( "         and start training again.")
        for local_entry in local.iterdir():
            LinuxCloud.__upload(connection,local_entry,remote/local_entry.name,update_info=update_info,commands=commands)
            if commands and commands.get_stop_command():
                return

    @staticmethod
    def __upload(connection,local : Path,remote : Path,update_info,commands : TrainCommands=None):
        if local.is_dir():
            LinuxCloud.__upload_dir(connection,local,remote,update_info,commands)
        else:
            LinuxCloud.__upload_file(connection,local,remote,update_info)

    @staticmethod
    def __download_dir(connection,local : Path,remote : Path,update_info,commands=None,config:CloudConfig=None):
        for remote_entry,info in update_info.items():
            if commands and commands.get_stop_command():
                return
            if config and not BaseCloud._filter_download(config,remote_entry):
                continue
            if info[2] != 'regular file':
                continue #update_info is recursive, so no need to go into directories

            local_entry=local / remote_entry.relative_to(remote)
            LinuxCloud.__download_file(connection,local=local_entry,remote=remote_entry,update_info=update_info)

    @staticmethod
    def __download_file(connection,local : Path, remote : Path,update_info):
        if remote in update_info:
            file_info=update_info[remote]
            if (local.exists()
                and local.stat().st_size == file_info[0]
                and local.stat().st_mtime >= file_info[1]):
                    return
        #else: file doesn't exist remotely, but error will be raised by connection.get()

        print(f'\n\ndownloading {local}...')
        connection.get(remote=remote.as_posix(),local=str(local))

    def delete_workspace(self):
        self.connection.run(f"rm -r {shlex.quote(self.config.workspace_dir)}",in_stream=False)
