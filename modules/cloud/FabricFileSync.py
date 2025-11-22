from pathlib import Path

from modules.cloud.BaseSSHFileSync import BaseSSHFileSync
from modules.util.config.CloudConfig import CloudConfig, CloudSecretsConfig

import fabric


class FabricFileSync(BaseSSHFileSync):
    def __init__(self, config: CloudConfig, secrets: CloudSecretsConfig):
        super().__init__(config,secrets)

    def __upload_batch(self,local_files,remote_dir : Path):
        with fabric.Connection(host=self.secrets.host,
                               port=self.secrets.port,
                               user=self.secrets.user,
                               connect_kwargs=self.secrets.connect_kwargs()) as connection:
            for local_file in local_files:
                self.__put(connection,local_file=local_file,remote_file=remote_dir / local_file.name)

    def upload_files(self,local_files,remote_dir : Path):
        if len(local_files) == 1:
            self.__put(self.sync_connection,local_file=local_files[0],remote_file=remote_dir/local_files[0].name)
        else:
            self._run_batches(
                lambda local_files:self.__upload_batch(local_files=local_files,remote_dir=remote_dir),
                tasks=local_files,
                workers=4,
                max_batch_size=100)

    def __download_batch(self,local_dir : Path,remote_files):
        with fabric.Connection(host=self.secrets.host,
                               port=self.secrets.port,
                               user=self.secrets.user,
                               connect_kwargs=self.secrets.connect_kwargs()) as connection:
            for remote_file in remote_files:
                self.__get(connection,local_file=local_dir / remote_file.name,remote_file=remote_file)

    def download_files(self,local_dir : Path,remote_files):
        if len(remote_files) == 1:
            self.__get(self.sync_connection,local_file=local_dir/remote_files[0].name,remote_file=remote_files[0])
        else:
            self._run_batches(
                lambda remote_files:self.__download_batch(local_dir=local_dir,remote_files=remote_files),
                tasks=remote_files,
                workers=4,
                max_batch_size=100)

    def upload_file(self,local_file: Path,remote_file: Path):
        self.__put(self.sync_connection,local_file,remote_file)

    def download_file(self,local_file: Path,remote_file: Path):
        self.__get(self.sync_connection,local_file,remote_file)

    @staticmethod
    def __put(connection,local_file: Path,remote_file: Path):
        print(f"Uploading {str(local_file)}...")
        try:
            connection.put(local=str(local_file),remote=remote_file.as_posix())
        except OSError: #https://github.com/paramiko/paramiko/issues/2484
            connection.close()
            raise

    @staticmethod
    def __get(connection,local_file: Path,remote_file: Path):
        print(f"\nDownloading {str(local_file)}...")
        try:
            connection.get(remote=remote_file.as_posix(),local=str(local_file))
        except OSError:
            connection.close()
            raise
