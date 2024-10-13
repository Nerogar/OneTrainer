from modules.cloud.LinuxCloud import LinuxCloud
from modules.util.config.TrainConfig import TrainConfig

import runpod
import time

class RunpodCloud(LinuxCloud):
    def __init__(self, config: TrainConfig):
        super(RunpodCloud, self).__init__(config)
        
        runpod.api_key=config.cloud.api_key

    def __get_host_port(self):
        config=self.config.cloud
        resumed=False
        while True:
            if (pod:=runpod.get_pod(config.id)) is None and not resumed: raise ValueError(f"Runpod {config.id} does not exist")
            if pod and pod['desiredStatus'] == "EXITED":
                self._start()
                #In edge cases runpod returns incorrect information for resumed pods:
                #The pod id seems to disappear for a while, and on recently stopped pods the old public IP and port is still being reported
                #Therefore, on resumed pods, iterate until there is a successful connection
                resumed=True
            elif pod and (runtime:=pod['runtime']) is not None:
                for port in runtime['ports']:
                    if port['isIpPublic']:
                        config.host=port['ip']
                        config.port=port['publicPort']
                        if resumed:
                            try: super()._connect()
                            except: continue
                        return
            print("waiting for public IP...")
            time.sleep(5)
    
        
    def _connect(self):
        if self.config.cloud.id != "": self.__get_host_port()
        super()._connect()
      
        
        
    def _create(self):
        config=self.config.cloud
        pod=runpod.create_pod(
            name=config.name,
            image_name="runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
            gpu_type_id=config.gpu_type,
            cloud_type=config.sub_type,
            support_public_ip=True,
            volume_in_gb=config.volume_size,
            container_disk_in_gb=10,
            ports="22/tcp",
            volume_mount_path="/workspace",
#            min_download=1000 # TODO parameter
        )
        config.id=pod['id']

    def delete(self):
        runpod.terminate_pod(self.config.cloud.id)

    def stop(self):
        runpod.stop_pod(self.config.cloud.id)
        
    def _start(self):
        runpod.resume_pod(self.config.cloud.id,gpu_count=1)
