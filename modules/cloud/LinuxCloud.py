import fabric
import shlex
import pickle

from modules.cloud.BaseCloud import BaseCloud
from modules.util.config.TrainConfig import TrainConfig
from modules.util.config.CloudConfig import CloudConfig
from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from modules.util.commands.TrainCommands import TrainCommands
from pathlib import Path


class LinuxCloud(BaseCloud):
    def __init__(self, config: TrainConfig):
        super(LinuxCloud, self).__init__(config)
        self.connection=None
        self.callback_connection=None
        self.callback_file=None
        self.command_connection=None
        self.command_pipe=None
        self.sync_connection=None

    def _connect(self):
        if self.connection: return
        config=self.config.cloud
        if config.host != '' and config.port != '':
            self.connection=fabric.Connection(host=config.host,port=config.port,user=config.user)
            self.connection.open()
            
            self.callback_connection=fabric.Connection(host=config.host,port=config.port,user=config.user)
            self.callback_connection.open()
            
            self.command_connection=fabric.Connection(host=config.host,port=config.port,user=config.user)
            self.command_connection.open()
            
            self.sync_connection=fabric.Connection(host=config.host,port=config.port,user=config.user)
            self.sync_connection.open()
        else: raise ValueError('Host and port required for SSH connection')
    
        
    def setup(self):
        super().setup()
        self.callback_file=f'{shlex.quote(self.config.cloud.workspace_dir)}/{self.config.cloud.run_id}.callback'
        self.command_pipe=f'{shlex.quote(self.config.cloud.workspace_dir)}/{self.config.cloud.run_id}.command'
        self.config_file=f'{shlex.quote(self.config.cloud.workspace_dir)}/{self.config.cloud.run_id}.json'
        self.connection.run(f'mkfifo {self.command_pipe}',warn=True,hide=True,in_stream=False)

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

    def close(self):
        if self.callback_connection: self.callback_connection.close()
        if self.command_connection: self.command_connection.close()
        if self.sync_connection: self.sync_connection.close()
        if self.connection: self.connection.close()

    def can_reattach(self):
        config=self.config.cloud
        result=self.connection.run(f"test -f {config.workspace_dir}/{config.run_id}.pid && test ! -f {config.workspace_dir}/{config.run_id}.finished",warn=True,in_stream=False)
        return result.exited == 0


    def run_trainer(self):
        config=self.config.cloud
        self.connection.put(f'scripts/train_remote.py',f'{config.onetrainer_dir}/scripts/train_remote.py') #TODO remove
        self.connection.put(f'modules/util/args/TrainArgs.py',f'{config.onetrainer_dir}/modules/util/args/TrainArgs.py') #TODO remove

        if self.can_reattach():
            print(f"Reattaching to run id {config.run_id}\n\n")
            self.__trail_detached_trainer()
            return
        

        cmd="export PYTHONUNBUFFERED=1"
        if config.huggingface_token != "": cmd+=f" && export HF_TOKEN={config.huggingface_token}"
        if config.huggingface_cache_dir != "": cmd+=f" && export HF_HOME={config.huggingface_cache_dir}"
        
        script_cmd=f'{config.onetrainer_dir}/run-cmd.sh train_remote --config-path={shlex.quote(self.config_file)} \
                                                                    --callback-path={self.callback_file} \
                                                                    --command-path={self.command_pipe}'


        if config.detach_trainer:
            cmd+=f' && nohup {script_cmd} >{config.workspace_dir}/{config.run_id}.out 2>&1 & echo $! > {config.workspace_dir}/{config.run_id}.pid'
            self.connection.run(cmd,disown=True)
            self.__trail_detached_trainer()
        else:
            cmd+=f' && {script_cmd}'
            self.connection.run(cmd,in_stream=False)

    def __trail_detached_trainer(self):
        config=self.config.cloud
        cmd=f'tail -f {config.workspace_dir}/{config.run_id}.out --pid $(<{config.workspace_dir}/{config.run_id}.pid)'
        self.connection.run(cmd,in_stream=False)
        #trainer has exited, don't reattach:
        self.connection.run(f'echo 1 > {self.config.cloud.workspace_dir}/{self.config.cloud.run_id}.finished')



                
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

        in_file,out_file,err_file=self.callback_connection.client.exec_command(cmd)
        
        try:
            while True:
                try: name=pickle.load(out_file)
                except EOFError: return
                params=pickle.load(out_file)

                fun=getattr(callbacks,name)
                fun(*params)
        finally:
            in_file.close()
            out_file.close()
            err_file.close()
        
    def send_commands(self,commands : TrainCommands):
        in_file,out_file,err_file=self.command_connection.client.exec_command(f'test -e {self.command_pipe} && cat > {self.command_pipe}')
        try:
            pickle.dump(commands,in_file)
            in_file.flush()
            in_file.channel.shutdown_write()
        finally:
            in_file.close()
            out_file.close()
            err_file.close()

        commands.reset()
        
    def _upload(self,local : Path,remote : Path,commands : TrainCommands = None):
        update_info=LinuxCloud.__get_update_info(self.connection,remote)
        LinuxCloud.__upload(self.connection,local,remote,update_info,commands=commands)

    def _upload_config_file(self,local : Path):
        self._upload(local,Path(self.config_file))


    def _download_file(self,local : Path,remote : Path):
        update_info=LinuxCloud.__get_update_info(self.connection,remote)
        LinuxCloud.__download_file(self.connection,local,remote,update_info)

    def sync_workspace(self):
        update_info=LinuxCloud.__get_update_info(self.sync_connection,Path(self.config.workspace_dir))
        LinuxCloud.__download_dir(self.sync_connection,
                                  local=Path(self.config.local_workspace_dir),
                                  remote=Path(self.config.workspace_dir),
                                  update_info=update_info,
                                  config=self.config.cloud)

    @staticmethod
    def __get_update_info(connection,remote : Path):
        cmd = f'find {shlex.quote(remote.as_posix())} -exec stat --printf "%n\\t%s\\t%Y\\t%F\\n"' + ' {} \\;'
        result=connection.run(cmd,warn=True,hide=True,in_stream=False)
        # dict[filename]: (size,mtime,filetype)
        ret={}
        for line in result.stdout.splitlines():
            sp=line.split('\t')
            ret[Path(sp[0])]=(int(sp[1]),int(sp[2]),sp[3])
        return ret;

    @staticmethod
    def __upload_file(connection,local : Path,remote : Path,update_info,commands=None):
        if remote in update_info:
            if (local.stat().st_size == update_info[remote][0]
                and local.stat().st_mtime <= update_info[remote][1]): return

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
            if commands and commands.get_stop_command(): return

    @staticmethod
    def __upload(connection,local : Path,remote : Path,update_info,commands : TrainCommands=None):
        if local.is_dir(): LinuxCloud.__upload_dir(connection,local,remote,update_info,commands)
        else: LinuxCloud.__upload_file(connection,local,remote,update_info)

    @staticmethod
    def __download_dir(connection,local : Path,remote : Path,update_info,commands=None,config:CloudConfig=None):
        for remote_entry,info in update_info.items():
            if commands and commands.get_stop_command(): return
            if config and not BaseCloud._filter_download(config,remote_entry): continue
            if not info[2] == 'regular file': continue #update_info is recursive, so no need to go into directories

            local_entry=local / remote_entry.relative_to(remote)
            LinuxCloud.__download_file(connection,local=local_entry,remote=remote_entry,update_info=update_info)
            
    @staticmethod
    def __download_file(connection,local : Path, remote : Path,update_info):
        if remote in update_info:
            file_info=update_info[remote]
            if (local.exists()
                and local.stat().st_size == file_info[0]
                and local.stat().st_mtime >= file_info[1]): return
        #else: file doesn't exist remotely, but error will be raised by connection.get()
        
        print(f'\n\ndownloading {local}...')
        connection.get(remote=remote.as_posix(),local=str(local))
