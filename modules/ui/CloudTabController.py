
import webbrowser

from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.CloudType import CloudType


class CloudTabController:
    def __init__(self, config: TrainConfig, parent):
        self.config = config
        self.parent = parent
        self.reattach = False

    def do_reattach(self):
        self.reattach = True
        try:
            self.parent.start_training()
        finally:
            self.reattach = False

    def get_gpu_types(self) -> list[str]:
        if self.config.cloud.type == CloudType.RUNPOD:
            import runpod
            runpod.api_key = self.config.secrets.cloud.api_key
            gpus = runpod.get_gpus()
            return [gpu['id'] for gpu in gpus]
        return []

    def open_create_cloud_url(self):
        if self.config.cloud.type == CloudType.RUNPOD:
            webbrowser.open("https://www.runpod.io/console/deploy?template=1a33vbssq9&type=gpu", new=0, autoraise=False)
