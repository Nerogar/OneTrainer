from modules.ui.controllers.BaseController import BaseController
from modules.ui.models.StateModel import StateModel
from modules.ui.models.TrainingModel import TrainingModel
from modules.ui.utils.WorkerPool import WorkerPool
from modules.util.enum.CloudAction import CloudAction
from modules.util.enum.CloudFileSync import CloudFileSync
from modules.util.enum.CloudSubtype import CloudSubtype
from modules.util.enum.CloudType import CloudType

import PySide6.QtWidgets as QtW
from PySide6.QtCore import QCoreApplication as QCA
from PySide6.QtCore import Slot


class CloudController(BaseController):
    state_ui_connections = {
        "cloud.enabled": "enabledCbx",
        "cloud.type": "cloudTypeCmb",
        "cloud.file_sync": "fileSyncMethodCmb",
        "secrets.cloud.api_key": "apiKeyLed",
        "secrets.cloud.host": "hostnameLed",
        "secrets.cloud.port": "portSbx",
        "secrets.cloud.user": "userLed",
        "secrets.cloud.id": "cloudIdLed",
        "cloud.tensorboard_tunnel": "tensorboardTcpTunnelCbx",
        "cloud.detach_trainer": "detachRemoteTrainerCbx",
        "cloud.run_id": "reattachIdLed",
        "cloud.download_samples": "downloadSamplesCbx",
        "cloud.download_output_model": "downloadOutputModelCbx",
        "cloud.download_saves": "downloadSavedCheckpointsCbx",
        "cloud.download_backups": "downloadBackupsCbx",
        "cloud.download_tensorboard": "downloadTensorboardLogCbx",
        "cloud.delete_workspace": "deleteRemoteWorkspaceCbx",
        "cloud.remote_dir": "remoteDirectoryLed",
        "cloud.onetrainer_dir": "onetrainerDirectoryLed",
        "cloud.huggingface_cache_dir": "huggingfaceCacheLed",
        "cloud.install_onetrainer": "installOnetrainerCbx",
        "cloud.install_cmd": "installCommandLed",
        "cloud.update_onetrainer": "updateOnetrainerCbx",
        "cloud.create": "createCloudCbx",
        "cloud.name": "cloudNameLed",
        "cloud.sub_type": "subTypeCmb",
        "cloud.gpu_type": "gpuCmb",
        "cloud.volume_size": "volumeSizeSbx",
        "cloud.min_download": "minDownloadSbx",
        "cloud.on_finish": "onFinishCmb",
        "cloud.on_error": "onErrorCmb",
        "cloud.on_detached_finish": "onDetachedCmb",
        "cloud.on_detached_error": "onDetachedErrorCmb",
    }

    def __init__(self, loader, parent=None):
        super().__init__(loader, "modules/ui/views/tabs/cloud.ui", name=QCA.translate("main_window_tabs", "Cloud"), parent=parent)



    ###FSM###

    def _connectUIBehavior(self):
        self._connect(self.ui.createCloudBtn.clicked, self.__createCloud())
        self._connect(self.ui.gpuBtn.clicked, self.__getGPUTypes())
        self._connect(self.ui.reattachBtn.clicked, self.__reattach())

        self._connect([self.ui.enabledCbx.toggled, QtW.QApplication.instance().stateChanged],
                      self.__enableCloud(), update_after_connect=True)



    def _loadPresets(self):
        for ctl in [self.ui.onFinishCmb, self.ui.onErrorCmb, self.ui.onDetachedCmb, self.ui.onDetachedErrorCmb]:
            for e in CloudAction.enabled_values():
                ctl.addItem(e.pretty_print(), userData=e)

        for e in CloudType.enabled_values():
            self.ui.cloudTypeCmb.addItem(e.pretty_print(), userData=e)

        for e in CloudFileSync.enabled_values():
            self.ui.fileSyncMethodCmb.addItem(e.pretty_print(), userData=e)

        for e in CloudSubtype.enabled_values():
            self.ui.subTypeCmb.addItem(e.pretty_print(), userData=e)

    ###Reactions###

    def __reattach(self):
        @Slot()
        def f():
            worker, name = WorkerPool.instance().createNamed(self.__train(), "train", poolless=True, daemon=True)
            if worker is not None:
                worker.connectCallbacks(init_fn=self.__enableReattach(False), result_fn=None,
                               finished_fn=self.__enableReattach(True),
                               errored_fn=self.__enableReattach(True), aborted_fn=self.__enableReattach(True))
                WorkerPool.instance().start(name)

        return f

    def __enableReattach(self, enabled):
        @Slot()
        def f():
            self.ui.reattachBtn.setEnabled(enabled)
        return f

    def __enableCloud(self):
        @Slot()
        def f():
            self.ui.frame.setEnabled(self.ui.enabledCbx.isChecked())
        return f


    def __getGPUTypes(self):
        @Slot()
        def f():
            self.ui.gpuCmb.clear()
            for gpu in StateModel.instance().get_gpus():
                self.ui.gpuCmb.addItem(gpu.name, userData=gpu)

        return f

    def __createCloud(self):
        @Slot()
        def f():
            if StateModel.instance().get_state("cloud.type") == CloudType.RUNPOD:
                self._openUrl("https://www.runpod.io/console/deploy?template=1a33vbssq9&type=gpu")
        return f

    ###Utils###

    def __train(self):
        def f(progress_fn=None):
            TrainingModel.instance().train(reattach=True, progress_fn=progress_fn)
        return f
