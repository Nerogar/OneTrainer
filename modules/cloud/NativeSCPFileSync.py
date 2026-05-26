import subprocess
from pathlib import Path

from modules.cloud.BaseSSHFileSync import BaseSSHFileSync
from modules.util.config.CloudConfig import CloudConfig, CloudSecretsConfig


class NativeSCPFileSync(BaseSSHFileSync):
    def __init__(self, config: CloudConfig, secrets: CloudSecretsConfig):
        super().__init__(config, secrets)
        password = getattr(secrets, "password", "").strip()
        if password:
            # Requires sshpass to be installed locally, will error if not
            self.base_args = ["sshpass", "-p", password, "scp"]
        else:
            self.base_args = ["scp"]
        key_file=secrets.expanded_key_file()
        if key_file:
            self.base_args.extend(["-i", key_file])
        self.base_args.extend([
                "-P", str(secrets.port),
                "-o", "StrictHostKeyChecking=no",
            ])

    def __upload_batch(self,local_files,remote_dir : Path):
        args=self.base_args.copy()
        args.extend(str(file) for file in local_files)
        args.append(f"{self.secrets.user}@{self.secrets.host}:{remote_dir.as_posix()}")
        subprocess.run(args).check_returncode()

    def upload_files(self,local_files,remote_dir : Path):
        self._run_batches(
            lambda local_files:self.__upload_batch(local_files=local_files,remote_dir=remote_dir),
            tasks=local_files,
            workers=4,
            max_batch_size=50)

    def __download_batch(self,local_dir : Path,remote_files):
        args=self.base_args.copy()
        args.extend(f"{self.secrets.user}@{self.secrets.host}:{file.as_posix()}" for file in remote_files)
        args.append(local_dir)
        subprocess.run(args).check_returncode()

    def download_files(self,local_dir : Path,remote_files):
        self._run_batches(
            lambda remote_files:self.__download_batch(local_dir=local_dir,remote_files=remote_files),
            tasks=remote_files,
            workers=4,
            max_batch_size=50)

    def upload_file(self,local_file: Path,remote_file: Path):
        subprocess.run(self.base_args + [
                str(local_file),
                f"{self.secrets.user}@{self.secrets.host}:{remote_file.as_posix()}",
            ]).check_returncode()

    def download_file(self,local_file: Path,remote_file: Path):
        subprocess.run(self.base_args + [
                f"{self.secrets.user}@{self.secrets.host}:{remote_file.as_posix()}",
                str(local_file),
            ]).check_returncode()
