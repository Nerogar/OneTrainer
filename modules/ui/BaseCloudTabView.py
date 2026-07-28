
from abc import ABC, abstractmethod

from modules.util.enum.CloudAction import CloudAction
from modules.util.enum.CloudFileSync import CloudFileSync
from modules.util.enum.CloudType import CloudType


class BaseCloudTabView(ABC):
    def __init__(self, components, controller):
        self.components = components
        self.controller = controller

    @property
    def reattach(self):
        return self.controller.reattach

    @abstractmethod
    def _make_reattach_frame(self, frame): pass

    @abstractmethod
    def _make_create_frame(self, frame): pass

    @abstractmethod
    def _on_set_gpu_types(self): pass

    def build_content(self, frame, controller, ui_state):
        self.components.label(frame, 0, 0, "启用",
                         tooltip="启用云端训练")
        self.components.switch(frame, 0, 1, ui_state, "cloud.enabled")

        self.components.label(frame, 1, 0, "类型",
                         tooltip="LINUX通过SSH连接Linux机器，RUNPOD自动创建和删除Pod")
        self.components.options_kv(frame, 1, 1, [
            ("RUNPOD", CloudType.RUNPOD),
            ("LINUX", CloudType.LINUX),
        ], ui_state, "cloud.type")

        self.components.label(frame, 2, 0, "文件同步方式",
                         tooltip="NATIVE_SCP使用scp.exe传输，FABRIC_SFTP使用Paramiko SFTP")
        self.components.options_kv(frame, 2, 1, [
            ("NATIVE_SCP", CloudFileSync.NATIVE_SCP),
            ("FABRIC_SFTP", CloudFileSync.FABRIC_SFTP),
        ], ui_state, "cloud.file_sync")

        self.components.label(frame, 3, 0, "API密钥",
                         tooltip="RUNPOD云服务API密钥，LINUX留空。单独存储不保存到配置文件")
        self.components.entry(frame, 3, 1, ui_state, "secrets.cloud.api_key")

        self.components.label(frame, 4, 0, "主机名",
                         tooltip="SSH服务器主机名或IP，有Cloud ID时留空")
        self.components.entry(frame, 4, 1, ui_state, "secrets.cloud.host")

        self.components.label(frame, 5, 0, "端口",
                         tooltip="SSH服务器端口，有Cloud ID时留空")
        self.components.entry(frame, 5, 1, ui_state, "secrets.cloud.port")

        self.components.label(frame, 6, 0, "用户",
                         tooltip='SSH username. Use "root" for RUNPOD. Your SSH client must be set up to connect to the cloud using a public key, without a password. For RUNPOD, create an ed25519 key locally, and copy the contents of the public keyfile to your "SSH公钥" on the RunPod website.')
        self.components.entry(frame, 6, 1, ui_state, "secrets.cloud.user")

        self.components.label(frame, 7, 0, "SSH密钥路径",
                 tooltip="SSH私钥文件绝对路径，留空使用系统SSH配置")
        self.components.path_entry(frame, 7, 1, ui_state, "secrets.cloud.key_file", mode="file")

        self.components.label(frame, 8, 0, "SSH密码",
                         tooltip="SSH密码认证，留空使用密钥认证")
        self.components.entry(frame, 8, 1, ui_state, "secrets.cloud.password")

        self.components.label(frame, 9, 0, "云ID",
                         tooltip="RUNPOD云ID，需要有公网IP和SSH服务")
        self.components.entry(frame, 9, 1, ui_state, "secrets.cloud.id")

        self.components.label(frame, 10, 0, "Tensorboard TCP隧道",
                         tooltip="通过TCP隧道连接云端Tensorboard")
        self.components.switch(frame, 10, 1, ui_state, "cloud.tensorboard_tunnel")

        self.components.label(frame, 1, 2, "远程目录",
                         tooltip="云端上传下载文件的目录")
        self.components.entry(frame, 1, 3, ui_state, "cloud.remote_dir")
        self.components.label(frame, 2, 2, "OneTrainer目录",
                         tooltip="云端OneTrainer目录")
        self.components.entry(frame, 2, 3, ui_state, "cloud.onetrainer_dir")
        self.components.label(frame, 3, 2, "Hugging Face缓存目录",
                         tooltip="Huggingface模型下载到此远程目录")
        self.components.entry(frame, 3, 3, ui_state, "cloud.huggingface_cache_dir")
        self.components.label(frame, 4, 2, "安装OneTrainer",
                         tooltip="如果目录不存在，自动从GitHub安装OneTrainer")
        self.components.switch(frame, 4, 3, ui_state, "cloud.install_onetrainer")
        self.components.label(frame, 5, 2, "安装命令",
                         tooltip="OneTrainer安装命令，默认即可")
        self.components.entry(frame, 5, 3, ui_state, "cloud.install_cmd")
        self.components.label(frame, 6, 2, "更新OneTrainer",
                         tooltip="如果云端已存在则更新OneTrainer")
        self.components.switch(frame, 6, 3, ui_state, "cloud.update_onetrainer")

        self.components.label(frame, 8, 2, "断开远程训练器",
                         tooltip="允许训练器在断连后继续运行")
        self.components.switch(frame, 8, 3, ui_state, "cloud.detach_trainer")
        self.components.label(frame, 9, 2, "重连ID",
                         tooltip="远程训练器标识ID，断连后可重新连接")
        reattach_frame = self._make_reattach_frame(frame)
        self.components.entry(reattach_frame, 0, 0, ui_state, "cloud.run_id", width=60)
        self.components.button(reattach_frame, 0, 1, "立即重连", controller.do_reattach)

        self.components.label(frame, 11, 2, "下载样本",
                         tooltip="从远程下载样本到本地")
        self.components.switch(frame, 11, 3, ui_state, "cloud.download_samples")
        self.components.label(frame, 12, 2, "下载输出模型",
                         tooltip="训练后下载最终模型")
        self.components.switch(frame, 12, 3, ui_state, "cloud.download_output_model")
        self.components.label(frame, 13, 2, "下载已保存检查点",
                         tooltip="从远程下载训练检查点到本地")
        self.components.switch(frame, 13, 3, ui_state, "cloud.download_saves")
        self.components.label(frame, 14, 2, "下载备份",
                         tooltip="从远程下载备份到本地")
        self.components.switch(frame, 14, 3, ui_state, "cloud.download_backups")
        self.components.label(frame, 15, 2, "下载Tensorboard日志",
                         tooltip="从远程下载Tensorboard日志到本地查看")
        self.components.switch(frame, 15, 3, ui_state, "cloud.download_tensorboard")
        self.components.label(frame, 16, 2, "删除远程工作空间",
                         tooltip="训练完成并下载数据后删除云端工作空间目录")
        self.components.switch(frame, 16, 3, ui_state, "cloud.delete_workspace")

        self.components.label(frame, 1, 4, "通过API创建云",
                         tooltip="主机和云ID为空时自动创建云实例，目前支持RUNPOD")
        create_frame = self._make_create_frame(frame)
        self.components.switch(create_frame, 0, 0, ui_state, "cloud.create")
        self.components.button(create_frame, 0, 1, "通过网站创建云", controller.open_create_cloud_url)

        self.components.label(frame, 2, 4, "云名称",
                         tooltip="新云实例名称")
        self.components.entry(frame, 2, 5, ui_state, "cloud.name")
        self.components.label(frame, 3, 4, "类型",
                         tooltip="选择RunPod云类型，详见RunPod网站")
        self.components.options_kv(frame, 3, 5, [
            ("", ""),
            ("社区", "COMMUNITY"),
            ("安全", "SECURE"),
        ], ui_state, "cloud.sub_type")

        self.components.label(frame, 4, 4, "GPU",
                         tooltip="选择GPU类型，请先输入API密钥")
        _, gpu_components = self.components.options_adv(frame, 4, 5, [("")], ui_state, "cloud.gpu_type", adv_command=self._on_set_gpu_types)
        self.gpu_types_menu = gpu_components['component']

        self.components.label(frame, 5, 4, "卷大小",
                         tooltip="存储卷大小(GB)，云删除后不保留")
        self.components.entry(frame, 5, 5, ui_state, "cloud.volume_size")

        self.components.label(frame, 6, 4, "最小下载速度",
                         tooltip="云端最小下载速度(Mbps)")
        self.components.entry(frame, 6, 5, ui_state, "cloud.min_download")

        self.components.label(frame, 8, 4, "完成时操作",
                         tooltip="训练完成且数据下载后的操作")
        self.components.options_kv(frame, 8, 5, [
            ("无", CloudAction.NONE),
            ("Stop", CloudAction.STOP),
            ("删除", CloudAction.DELETE),
        ], ui_state, "cloud.on_finish")

        self.components.label(frame, 9, 4, "错误时操作",
                         tooltip="训练出错时的操作，数据可能丢失")
        self.components.options_kv(frame, 9, 5, [
            ("无", CloudAction.NONE),
            ("Stop", CloudAction.STOP),
            ("删除", CloudAction.DELETE),
        ], ui_state, "cloud.on_error")

        self.components.label(frame, 10, 4, "断连完成时操作",
                         tooltip="训练完成但客户端已断连时的操作")
        self.components.options_kv(frame, 10, 5, [
            ("无", CloudAction.NONE),
            ("Stop", CloudAction.STOP),
            ("删除", CloudAction.DELETE),
        ], ui_state, "cloud.on_detached_finish")

        self.components.label(frame, 11, 4, "断连错误时操作",
                         tooltip="训练出错且客户端已断连时的操作")
        self.components.options_kv(frame, 11, 5, [
            ("无", CloudAction.NONE),
            ("Stop", CloudAction.STOP),
            ("删除", CloudAction.DELETE),
        ], ui_state, "cloud.on_detached_error")
