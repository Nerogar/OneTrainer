import shlex
from abc import abstractmethod
from pathlib import Path

from modules.cloud.BaseFileSync import BaseFileSync
from modules.util.config.CloudConfig import CloudConfig, CloudSecretsConfig

import fabric


class BaseSSHFileSync(BaseFileSync):
    def __init__(self, config: CloudConfig, secrets: CloudSecretsConfig):
        super().__init__(config, secrets)
        self.sync_connection=fabric.Connection(host=secrets.host,
                               port=secrets.port,
                               user=secrets.user,
                               connect_kwargs=secrets.connect_kwargs())

    def close(self):
        if self.sync_connection:
            self.sync_connection.close()

    @abstractmethod
    def upload_files(self,local_files,remote_dir: Path):
        pass

    @abstractmethod
    def download_files(self,local_dir: Path,remote_files):
        pass

    @abstractmethod
    def upload_file(self,local_file: Path,remote_file: Path):
        pass

    @abstractmethod
    def download_file(self,local_file: Path,remote_file: Path):
        pass

    def sync_up_file(self,local : Path,remote : Path):
        sync_info=self.__get_sync_info(remote)
        if not self.__needs_upload(local=local,remote=remote,sync_info=sync_info):
            return

        self.sync_connection.open()
        self.sync_connection.run(f'mkdir -p {shlex.quote(remote.parent.as_posix())}',in_stream=False)
        self.upload_file(local_file=local,remote_file=remote)


    def sync_up_dir(self,local : Path,remote: Path,recursive: bool,sync_info=None):
        if sync_info is None:
            sync_info=self.__get_sync_info(remote)
        self.sync_connection.open()
        self.sync_connection.run(f'mkdir -p {shlex.quote(remote.as_posix())}',in_stream=False)
        files=[]
        for local_entry in local.iterdir():
            if local_entry.is_file():
                remote_entry=remote/local_entry.name
                if self.__needs_upload(local=local_entry,remote=remote_entry,sync_info=sync_info):
                    files.append(local_entry)
            elif recursive and local_entry.is_dir():
                self.sync_up_dir(local=local_entry,remote=remote/local_entry.name,recursive=True,sync_info=sync_info)

        self.upload_files(local_files=files,remote_dir=remote)

    def sync_down_file(self,local : Path,remote : Path):
        sync_info=self.__get_sync_info(remote)
        if not self.__needs_download(local=local,remote=remote,sync_info=sync_info):
            return
        local.parent.mkdir(parents=True,exist_ok=True)
        self.download_file(local_file=local,remote_file=remote)

    def sync_down_dir(self,local : Path,remote : Path,filter=None):
        sync_info=self.__get_sync_info(remote)
        dirs={}
        for remote_entry in sync_info:
            local_entry=local / remote_entry.relative_to(remote)
            if ((filter is not None and not filter(remote_entry))
                or not self.__needs_download(local=local_entry,remote=remote_entry,sync_info=sync_info)):
                continue

            if local_entry.parent not in dirs:
                dirs[local_entry.parent]=[]
            dirs[local_entry.parent].append(remote_entry)

        for dir,files in dirs.items():
            dir.mkdir(parents=True,exist_ok=True)
            self.download_files(local_dir=dir,remote_files=files)


    def __get_sync_info(self,remote : Path):
        cmd = f'find {shlex.quote(remote.as_posix())} -type f -exec stat --printf "%n\\t%s\\t%Y\\n"' + ' {} \\;'
        self.sync_connection.open()
        result=self.sync_connection.run(cmd,warn=True,hide=True,in_stream=False)
        info={}
        for line in result.stdout.splitlines():
            sp=line.split('\t')
            info[Path(sp[0])]={
                    'size': int(sp[1]),
                    'mtime': int(sp[2])
                }
        return info

    @staticmethod
    def __needs_upload(local : Path,remote : Path,sync_info):
        return (
            remote not in sync_info
            or local.stat().st_size != sync_info[remote]['size']
            or local.stat().st_mtime > sync_info[remote]['mtime']
        )

    @staticmethod
    def __needs_download(local : Path,remote : Path,sync_info):
        return (
            not local.exists()
            or remote not in sync_info
            or local.stat().st_size != sync_info[remote]['size']
            or local.stat().st_mtime < sync_info[remote]['mtime']
        )
