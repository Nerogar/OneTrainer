import subprocess
from pathlib import Path

from modules.cloud.BaseSSHFileSync import BaseSSHFileSync
from modules.util.config.CloudConfig import CloudConfig


class NativeSCPFileSync(BaseSSHFileSync):
    def __init__(self, config: CloudConfig):
        super().__init__(config)
        self.base_args=[
                "scp",
                "-P", str(config.port),
                "-o", "StrictHostKeyChecking=no",
            ]

    def __upload_batch(self,local_files,remote_dir : Path):
        args=self.base_args.copy()
        args.extend(str(file) for file in local_files)
        args.append(f"{self.config.user}@{self.config.host}:{remote_dir.as_posix()}")
        subprocess.run(args).check_returncode()

    def upload_files(self,local_files,remote_dir : Path):
        self._run_batches(
            lambda local_files:self.__upload_batch(local_files=local_files,remote_dir=remote_dir),
            tasks=local_files,
            workers=4,
            max_batch_size=50)

    def __download_batch(self,local_dir : Path,remote_files):
        args=self.base_args.copy()
        args.extend(f"{self.config.user}@{self.config.host}:{file.as_posix()}" for file in remote_files)
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
                f"{self.config.user}@{self.config.host}:{remote_file.as_posix()}",
            ]).check_returncode()

    def download_file(self,local_file: Path,remote_file: Path):
        subprocess.run(self.base_args + [
                f"{self.config.user}@{self.config.host}:{remote_file.as_posix()}",
                str(local_file),
            ]).check_returncode()
