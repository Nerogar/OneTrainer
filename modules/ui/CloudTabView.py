from abc import ABC, abstractmethod

from modules.ui.CloudTabController import CloudTabController
from modules.util.enum.CloudAction import CloudAction
from modules.util.enum.CloudFileSync import CloudFileSync
from modules.util.enum.CloudType import CloudType
from modules.util.ui import pyside6_components
from modules.util.ui.pyside6_util import QtABCMeta, fit_form

from PySide6.QtWidgets import QWidget


class BaseCloudTabView(ABC):
    def __init__(self, components, controller):
        self.components = components
        self.controller = controller

    @property
    def reattach(self):
        return self.controller.reattach

    @abstractmethod
    def _on_set_gpu_types(self): pass

    def _build_connection(self, frame, controller, ui_state):
        self.components.label(frame, 0, 0, "Enabled",
                         tooltip="Enable cloud training")
        self.components.switch(frame, 0, 1, ui_state, "cloud.enabled")

        self.components.label(frame, 1, 0, "Type",
                         tooltip="Choose LINUX to connect to a linux machine via SSH. Choose RUNPOD for additional functionality such as automatically creating and deleting pods.")
        self.components.options_kv(frame, 1, 1, [
            ("RUNPOD", CloudType.RUNPOD),
            ("LINUX", CloudType.LINUX),
        ], ui_state, "cloud.type")

        self.components.label(frame, 2, 0, "File sync method",
                         tooltip="Choose NATIVE_SCP to use scp.exe to transfer files. FABRIC_SFTP uses the Paramiko/Fabric SFTP implementation for file transfers instead.")
        self.components.options_kv(frame, 2, 1, [
            ("NATIVE_SCP", CloudFileSync.NATIVE_SCP),
            ("FABRIC_SFTP", CloudFileSync.FABRIC_SFTP),
        ], ui_state, "cloud.file_sync")

        self.components.label(frame, 3, 0, "API key",
                         tooltip="Cloud service API key for RUNPOD. Leave empty for LINUX. This value is stored separately, not saved to your configuration file. ")
        self.components.entry(frame, 3, 1, ui_state, "secrets.cloud.api_key")

        self.components.label(frame, 4, 0, "Hostname",
                         tooltip="SSH server hostname or IP. Leave empty if you have a Cloud ID or want to automatically create a new cloud.")
        self.components.entry(frame, 4, 1, ui_state, "secrets.cloud.host")

        self.components.label(frame, 5, 0, "Port",
                         tooltip="SSH server port. Leave empty if you have a Cloud ID or want to automatically create a new cloud.")
        self.components.entry(frame, 5, 1, ui_state, "secrets.cloud.port", max_width=6)

        self.components.label(frame, 6, 0, "User",
                         tooltip='SSH username. Use "root" for RUNPOD. Your SSH client must be set up to connect to the cloud using a public key, without a password. For RUNPOD, create an ed25519 key locally, and copy the contents of the public keyfile to your "SSH Public Keys" on the RunPod website.')
        self.components.entry(frame, 6, 1, ui_state, "secrets.cloud.user")

        self.components.label(frame, 7, 0, "SSH keyfile path",
                 tooltip="Absolute path to the private key file used for SSH connections. Leave empty to rely on your system SSH configuration.")
        self.components.path_entry(frame, 7, 1, ui_state, "secrets.cloud.key_file", mode="file")

        self.components.label(frame, 8, 0, "SSH password",
                         tooltip="SSH password for password-based authentication. If you try to use native SCP requires sshpass to be installed. Leave empty to use key-based authentication.")
        self.components.entry(frame, 8, 1, ui_state, "secrets.cloud.password")

        self.components.label(frame, 9, 0, "Cloud id",
                         tooltip="RUNPOD Cloud ID. The cloud service must have a public IP and SSH service. Leave empty if you want to automatically create a new RUNPOD cloud, or if you're connecting to another cloud provider via SSH Hostname and Port.")
        self.components.entry(frame, 9, 1, ui_state, "secrets.cloud.id")

        self.components.label(frame, 10, 0, "Tensorboard TCP tunnel",
                         tooltip="Instead of starting tensorboard locally, make a TCP tunnel to a tensorboard on the cloud")
        self.components.switch(frame, 10, 1, ui_state, "cloud.tensorboard_tunnel")

        self.components.label(frame, 11, 0, "Detach remote trainer",
                         tooltip="Allows the trainer to keep running even if your connection to the cloud is lost.")
        self.components.switch(frame, 11, 1, ui_state, "cloud.detach_trainer")

        self.components.label(frame, 12, 0, "Reattach id",
                         tooltip="An id identifying the remotely running trainer. In case you have lost connection or closed OneTrainer, it will try to reattach to this id instead of starting a new remote trainer.")
        reattach_frame = self.components.inline_frame(frame, 12, 1)
        self.components.entry(reattach_frame, 0, 0, ui_state, "cloud.run_id", width=60)
        self.components.button(reattach_frame, 0, 1, "Reattach now", controller.do_reattach, sticky="nw")
        self.reattach_frame = reattach_frame

    def _build_setup(self, frame, controller, ui_state):
        self.components.label(frame, 0, 0, "Remote Directory",
                         tooltip="The directory on the cloud where files will be uploaded and downloaded.")
        self.components.entry(frame, 0, 1, ui_state, "cloud.remote_dir")

        self.components.label(frame, 1, 0, "OneTrainer Directory",
                         tooltip="The directory for OneTrainer on the cloud.")
        self.components.entry(frame, 1, 1, ui_state, "cloud.onetrainer_dir")

        self.components.label(frame, 2, 0, "Huggingface cache Directory",
                         tooltip="Huggingface models are downloaded to this remote directory.")
        self.components.entry(frame, 2, 1, ui_state, "cloud.huggingface_cache_dir")

        self.components.label(frame, 3, 0, "Install OneTrainer",
                         tooltip="Automatically install OneTrainer from GitHub if the directory doesn't already exist.")
        self.components.switch(frame, 3, 1, ui_state, "cloud.install_onetrainer")

        self.components.label(frame, 4, 0, "Install command",
                         tooltip="The command for installing OneTrainer. Leave the default, unless you want to use a development branch of OneTrainer.")
        self.components.entry(frame, 4, 1, ui_state, "cloud.install_cmd")

        self.components.label(frame, 5, 0, "Update OneTrainer",
                         tooltip="Update OneTrainer if it already exists on the cloud.")
        self.components.switch(frame, 5, 1, ui_state, "cloud.update_onetrainer")

    def _build_download(self, frame, controller, ui_state):
        self.components.label(frame, 0, 0, "Download samples",
                         tooltip="Download samples from the remote workspace directory to your local machine.")
        self.components.switch(frame, 0, 1, ui_state, "cloud.download_samples")

        self.components.label(frame, 1, 0, "Download output model",
                         tooltip="Download the final model after training to the local Final Output Directory, or to the local models/ folder for a cloud: destination. You can disable this if you plan to use an automatically saved checkpoint instead.")
        self.components.switch(frame, 1, 1, ui_state, "cloud.download_output_model")

        self.components.label(frame, 2, 0, "Download saved checkpoints",
                         tooltip="Download the automatically saved training checkpoints from the remote workspace directory to your local machine.")
        self.components.switch(frame, 2, 1, ui_state, "cloud.download_saves")

        self.components.label(frame, 3, 0, "Download backups",
                         tooltip="Download backups from the remote workspace directory to your local machine. It's usually not necessary to download them, because as long as the backups are still available on the cloud, the training can be restarted using one of the cloud's backups.")
        self.components.switch(frame, 3, 1, ui_state, "cloud.download_backups")

        self.components.label(frame, 4, 0, "Download tensorboard logs",
                         tooltip="Download TensorBoard event logs from the remote workspace directory to your local machine. They can then be viewed locally in TensorBoard. It is recommended to disable \"Sample to TensorBoard\" to reduce the event log size.")
        self.components.switch(frame, 4, 1, ui_state, "cloud.download_tensorboard")

    def _build_container(self, frame, controller, ui_state):
        self.components.label(frame, 0, 0, "Create cloud via API",
                         tooltip="Automatically creates a new cloud instance if both Host:Port and Cloud ID are empty. Currently supported for RUNPOD.")
        create_frame = self.components.inline_frame(frame, 0, 1)
        self.components.switch(create_frame, 0, 0, ui_state, "cloud.create")
        self.components.button(create_frame, 0, 1, "Create cloud via website", controller.open_create_cloud_url, sticky="nw")
        self.create_frame = create_frame

        self.components.label(frame, 1, 0, "Cloud name",
                         tooltip="The name of the new cloud instance.")
        self.components.entry(frame, 1, 1, ui_state, "cloud.name")

        self.components.label(frame, 2, 0, "Type",
                         tooltip="Select the RunPod cloud type. See RunPod's website for details.")
        self.components.options_kv(frame, 2, 1, [
            ("", ""),
            ("Community", "COMMUNITY"),
            ("Secure", "SECURE"),
        ], ui_state, "cloud.sub_type")

        self.components.label(frame, 3, 0, "GPU",
                         tooltip="Select the GPU type. Enter an API key before pressing the button.")
        _, gpu_components = self.components.options_adv(frame, 3, 1, [("")], ui_state, "cloud.gpu_type", adv_command=self._on_set_gpu_types)
        self.gpu_types_menu = gpu_components['component']

        self.components.label(frame, 4, 0, "Volume size (GB)",
                         tooltip="Set the storage volume size in GB. This volume persists only until the cloud is deleted - not a RunPod network volume")
        self.components.entry(frame, 4, 1, ui_state, "cloud.volume_size", max_width=7)

        self.components.label(frame, 5, 0, "Min download",
                         tooltip="Set the minimum download speed of the cloud in Mbps.")
        self.components.entry(frame, 5, 1, ui_state, "cloud.min_download", max_width=7)

    def _build_completion_actions(self, frame, controller, ui_state):
        self.components.label(frame, 0, 0, "Delete remote workspace",
                         tooltip="Delete the workspace directory on the cloud after training has finished successfully and data has been downloaded.")
        self.components.switch(frame, 0, 1, ui_state, "cloud.delete_workspace")

        self.components.label(frame, 1, 0, "Action on finish",
                         tooltip="What to do when training finishes and the data has been fully downloaded: Stop or delete the cloud, or do nothing.")
        self.components.options_kv(frame, 1, 1, [
            ("None", CloudAction.NONE),
            ("Stop", CloudAction.STOP),
            ("Delete", CloudAction.DELETE),
        ], ui_state, "cloud.on_finish")

        self.components.label(frame, 2, 0, "Action on error",
                         tooltip="What to do if training stops due to an error: Stop or delete the cloud, or do nothing. Data may be lost.")
        self.components.options_kv(frame, 2, 1, [
            ("None", CloudAction.NONE),
            ("Stop", CloudAction.STOP),
            ("Delete", CloudAction.DELETE),
        ], ui_state, "cloud.on_error")

        self.components.label(frame, 3, 0, "Action on detached finish",
                         tooltip="What to do when training finishes, but the client has been detached and cannot download data. Data may be lost.")
        self.components.options_kv(frame, 3, 1, [
            ("None", CloudAction.NONE),
            ("Stop", CloudAction.STOP),
            ("Delete", CloudAction.DELETE),
        ], ui_state, "cloud.on_detached_finish")

        self.components.label(frame, 4, 0, "Action on detached error",
                         tooltip="What to if training stops due to an error, but the client has been detached and cannot download data. Data may be lost.")
        self.components.options_kv(frame, 4, 1, [
            ("None", CloudAction.NONE),
            ("Stop", CloudAction.STOP),
            ("Delete", CloudAction.DELETE),
        ], ui_state, "cloud.on_detached_error")


class CloudTabView(BaseCloudTabView, QWidget, metaclass=QtABCMeta):

    def __init__(self, master, controller: CloudTabController, ui_state):
        QWidget.__init__(self, master)
        BaseCloudTabView.__init__(self, pyside6_components, controller)

        self.ui_state = ui_state
        layout = self.components._layout(self)
        layout.setRowStretch(0, 1)
        layout.setColumnStretch(0, 1)
        scroll, frame = self.components.scrollable_frame(self)
        layout.addWidget(scroll, 0, 0)
        self.frame = frame

        layout = self.components._layout(frame)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)
        left = self.components.inline_frame(frame, 0, 0)
        right = self.components.inline_frame(frame, 0, 1)
        for column in (left, right):
            self.components._layout(column).setColumnStretch(0, 1)

        for column, sections in (
            (left, (("Connection", self._build_connection), ("Setup", self._build_setup))),
            (right, (("Downloads", self._build_download), ("Container", self._build_container),
                     ("Completion Actions", self._build_completion_actions))),
        ):
            for row, (title, builder) in enumerate(sections):
                section = self.components.form_section(column, row, 0, title)
                builder(section, controller, ui_state)
                if title in ("Download", "Completion Actions"):
                    self.components._layout(section).setColumnStretch(1, 0)
                fit_form(section)
            self.components._pack_form(column)

        self.components._layout(self.reattach_frame).setColumnStretch(0, 1)
        self.components._layout(self.create_frame).setColumnStretch(2, 1)
        gpu_frame = self.gpu_types_menu.parentWidget()
        self.components._layout(gpu_frame).setColumnStretch(0, 0)
        self.components._pack_form(gpu_frame)
        self.components._pack_form(frame)

    def _on_set_gpu_types(self):
        self.gpu_types_menu.clear()
        self.gpu_types_menu.addItems(self.controller.get_gpu_types())
        # The choices arrive after the form was fitted; allow longer GPU names.
        self.gpu_types_menu.setFixedWidth(self.gpu_types_menu.sizeHint().width())
