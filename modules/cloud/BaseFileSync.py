
import concurrent.futures
from abc import ABCMeta, abstractmethod
from math import ceil
from pathlib import Path

from modules.util.config.CloudConfig import CloudConfig


class BaseFileSync(metaclass=ABCMeta):
    def __init__(self, config: CloudConfig):
        super().__init__()
        self.config = config


    def sync_up(self,local : Path,remote : Path):
        if local.is_dir():
            self.sync_up_dir(local_dir=local,remote_dir=remote,recursive=True)
        else:
            self.sync_up_file(local_file=local,remote_file=remote)

    @abstractmethod
    def sync_up_file(self,local : Path,remote : Path):
        pass

    @abstractmethod
    def sync_up_dir(self,local : Path,remote: Path,recursive: bool):
        pass

    @abstractmethod
    def sync_down_file(self,local : Path,remote : Path):
        pass

    @abstractmethod
    def sync_down_dir(self,local : Path,remote : Path,filter=None):
        pass

    @staticmethod
    def _run_batches(fn,tasks,workers:int,max_batch_size=None):
        futures=[]
        if len(tasks) == 0:
            return
        batch_size=ceil(len(tasks) / workers)
        if max_batch_size is not None:
            batch_size=min(max_batch_size,batch_size)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            for i in range(0,len(tasks),batch_size):
                batch=tasks[i:i+batch_size]
                futures.append(executor.submit(fn,batch))

        for future in futures:
            if (exception:=future.exception()):
                raise exception
