import secrets as pysecrets
import time

from modules.cloud.LinuxCloud import LinuxCloud
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.CloudAction import CloudAction

import runpod


class RunpodCloud(LinuxCloud):
    def __init__(self, config: TrainConfig):
        super().__init__(config)

        runpod.api_key=config.secrets.cloud.api_key

    def __get_host_port(self):
        secrets=self.config.secrets.cloud
        resumed=False
        while True:
            if (pod:=runpod.get_pod(secrets.id)) is None and not resumed:
                raise ValueError(f"Runpod {secrets.id} does not exist")
            if pod and pod['desiredStatus'] == "EXITED":
                self._start()
                #In edge cases runpod returns incorrect information for resumed pods:
                #The pod id seems to disappear for a while, and on recently stopped pods the old public IP and port is still being reported
                #Therefore, on resumed pods, iterate until there is a successful connection
                resumed=True
            elif pod and (runtime:=pod['runtime']) is not None and 'ports' in runtime and runtime['ports'] is not None:
                for port in runtime['ports']:
                    if port['isIpPublic']:
                        secrets.host=port['ip']
                        secrets.port=port['publicPort']
                        if resumed:
                            try:
                                super()._connect()
                            except Exception:
                                continue
                        return
            if secrets.id == "":
                print("waiting for public IP...")
            else:
                print(f"waiting for public IP... Status: https://www.runpod.io/console/pods?id={secrets.id}")
            time.sleep(5)


    def _connect(self):
        config=self.config.cloud
        secrets=self.config.secrets.cloud

        pod=None
        if secrets.id != "":
            pod=runpod.get_pod(secrets.id)
            if pod is None:
                raise ValueError(f"Runpod {secrets.id} does not exist")
        elif config.create:
            self._create()
            pod=runpod.get_pod(secrets.id)
            if pod is None:
                raise ValueError("Could not create cloud")

        if pod is not None:
            self.__get_host_port()
        super()._connect()

    def _create(self):
        config=self.config.cloud
        secrets=self.config.secrets.cloud
        pod=runpod.create_pod(
            name=config.name,
            image_name="",
            template_id="1a33vbssq9",
            gpu_type_id=config.gpu_type,
            cloud_type=config.sub_type,
            support_public_ip=True,
            volume_in_gb=config.volume_size,
            container_disk_in_gb=20,
            volume_mount_path="/workspace",
            min_download=config.min_download,
            env={"JUPYTER_PASSWORD": pysecrets.token_urlsafe(16)},
        )
        secrets.id=pod['id']

    def delete(self):
        runpod.terminate_pod(self.config.secrets.cloud.id)

    def stop(self):
        runpod.stop_pod(self.config.secrets.cloud.id)

    def _start(self):
        runpod.resume_pod(self.config.secrets.cloud.id,gpu_count=1)

    def _get_action_cmd(self,action : CloudAction):
        if action == CloudAction.STOP:
            return "source /etc/rp_environment && runpodctl stop pod $RUNPOD_POD_ID"
        elif action == CloudAction.DELETE:
            return "source /etc/rp_environment && runpodctl remove pod $RUNPOD_POD_ID"
        else:
            return ":"
