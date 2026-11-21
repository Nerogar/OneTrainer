from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path

from modules.ui.AdditionalEmbeddingsTabController import AdditionalEmbeddingsTabController
from modules.ui.AdditionalEmbeddingsTabView import AdditionalEmbeddingsTabView
from modules.ui.CaptionUIView import CaptionUIView
from modules.ui.CloudTabController import CloudTabController
from modules.ui.CloudTabView import CloudTabView
from modules.ui.ConceptTabController import ConceptTabController
from modules.ui.ConceptTabView import ConceptTabView
from modules.ui.ConvertModelUIView import ConvertModelUIView
from modules.ui.GeneralTabController import GeneralTabController
from modules.ui.GeneralTabView import GeneralTabView
from modules.ui.LoraTabController import LoraTabController
from modules.ui.LoraTabView import LoraTabView
from modules.ui.ModelTabController import ModelTabController
from modules.ui.ModelTabView import ModelTabView
from modules.ui.ProfilingWindowController import ProfilingWindowController
from modules.ui.ProfilingWindowView import ProfilingWindowView
from modules.ui.SampleWindowView import SampleWindowView
from modules.ui.SamplingTabController import SamplingTabController
from modules.ui.SamplingTabView import SamplingTabView
from modules.ui.TopBarController import TopBarController
from modules.ui.TopBarView import TopBarView
from modules.ui.TrainingTabController import TrainingTabController
from modules.ui.TrainingTabView import TrainingTabView
from modules.ui.TrainUIController import TrainUIController
from modules.ui.VideoToolUIView import VideoToolUIView
from modules.util import path_util
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.DataType import DataType
from modules.util.enum.ImageFormat import ImageFormat
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.ui import pyside6_components
from modules.util.ui.pyside6_settings import MainWindowSettings
from modules.util.ui.pyside6_util import QtABCMeta, fit_form
from modules.util.ui.PySide6UIState import PySide6UIState

from PySide6.QtCore import QSettings, QTimer
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QFileDialog, QGridLayout, QMainWindow, QMessageBox, QTabWidget, QWidget


class BaseTrainUIView(ABC):
    def __init__(self, components, controller, ui_state):
        self.components = components
        self.controller = controller
        self.ui_state = ui_state

    # --- Abstract callbacks (controller calls into view) ---

    @abstractmethod
    def on_update_status(self, status: str): pass

    @abstractmethod
    def on_training_started(self): pass

    @abstractmethod
    def on_training_stopped(self, error_caught: bool): pass

    @abstractmethod
    def on_training_stopping(self): pass

    @abstractmethod
    def on_update_progress(self, epoch_step: int, max_step: int, epoch: int, max_epoch: int, eta_str: str | None): pass

    @abstractmethod
    def schedule_on_main_thread(self, fn: Callable): pass

    @abstractmethod
    def get_cloud_reattach(self) -> bool: pass

    @abstractmethod
    def save_default(self): pass

    @abstractmethod
    def show_validation_errors(self, errors: list[str]): pass

    @abstractmethod
    def confirm(self, title: str, message: str) -> bool: pass

    @abstractmethod
    def wait_window(self, window): pass

    @abstractmethod
    def show_window(self, window): pass

    @abstractmethod
    def connect_window_closed(self, window, callback): pass

    def sync_cloud_secrets(self):
        # Called from training thread — defer to main thread
        self.schedule_on_main_thread(
            lambda: self.ui_state.get_var("secrets.cloud").update(self.controller.train_config.secrets.cloud)
        )

    def start_training(self):
        self.controller.start_training()

    def open_tensorboard(self):
        self.controller.open_tensorboard()

    def sample_now(self):
        self.controller.sample_now()

    def backup_now(self):
        self.controller.backup_now()

    def save_now(self):
        self.controller.save_now()

    @abstractmethod
    def open_dataset_tool(self): pass

    @abstractmethod
    def open_video_tool(self): pass

    @abstractmethod
    def open_convert_model_tool(self): pass

    @abstractmethod
    def open_sampling_tool(self): pass

    @abstractmethod
    def open_manual_sample_window(self): pass

    @abstractmethod
    def open_profiling_tool(self): pass

    @abstractmethod
    def export_training(self): pass

    @abstractmethod
    def generate_debug_package(self): pass

    # --- Content builders (components calls; called after frame creation) ---

    def build_bottom_bar_content(self, frame, status_frame, controller, ui_state):
        self.set_step_progress, self.set_epoch_progress = self.components.double_progress(frame, 0, 0, "step", "epoch")

        self.status_label = self.components.label(status_frame, 0, 0, "", pad=0,
                                             tooltip="Current status of the training run")
        self.eta_label = self.components.label(status_frame, 1, 0, "", pad=0)

        self.export_button = self.components.button(frame, 0, 3, "Export", self.export_training,
                                             padx=5, pady=(15, 0), sticky="vew",
                                             tooltip="Export the current configuration as a script to run without a UI")
        self.export_button.setProperty("compactButton", True)

        debug_button = self.components.button(frame, 0, 4, "Debug", self.generate_debug_package,
                                             padx=(5, 25), pady=(15, 0), sticky="vew",
                                             tooltip="Generate a zip file with config.json, debug_report.log and settings diff, use this to report bugs or issues")
        debug_button.setProperty("compactButton", True)

        tensorboard_button = self.components.button(frame, 0, 6, "Tensorboard", self.open_tensorboard,
                                                   width=100, padx=(0, 5), pady=(15, 0), sticky="vew")
        tensorboard_button.setObjectName("tensorboardButton")

        self.training_button = self.components.button(frame, 0, 7, "Start Training", self.start_training,
                                                 padx=(5, 20), pady=(15, 0), sticky="vew")


    def build_data_tab_content(self, frame, controller, ui_state):
        # aspect ratio bucketing
        self.components.label(frame, 0, 0, "Aspect Ratio Bucketing",
                         tooltip="Aspect ratio bucketing enables training on images with different aspect ratios")
        self.components.switch(frame, 0, 1, ui_state, "aspect_ratio_bucketing")

        # latent caching
        self.components.label(frame, 1, 0, "Latent Caching",
                         tooltip="Caching of intermediate training data that can be re-used between epochs")
        self.components.switch(frame, 1, 1, ui_state, "latent_caching")

        # clear cache before training
        self.components.label(frame, 2, 0, "Clear cache before training",
                         tooltip="Clears the cache directory before starting to train. Only disable this if you want to continue using the same cached data. Disabling this can lead to errors, if other settings are changed during a restart")
        self.components.switch(frame, 2, 1, ui_state, "clear_cache_before_training")

    def build_sampling_tab_header(self, top_frame, sub_frame, controller, ui_state):
        self.components.label(top_frame, 0, 0, "Sample After",
                         tooltip="The interval used when automatically sampling from the model during training")
        self.components.time_entry(top_frame, 0, 1, ui_state, "sample_after", "sample_after_unit")

        self.components.label(top_frame, 0, 2, "Skip First",
                         tooltip="Start sampling automatically after this interval has elapsed.")
        self.components.entry(top_frame, 0, 3, ui_state, "sample_skip_first", width=50, sticky="nw")

        self.components.label(top_frame, 0, 4, "Format",
                         tooltip="File Format used when saving samples")
        self.components.options_kv(top_frame, 0, 5, [
            ("PNG", ImageFormat.PNG),
            ("JPG", ImageFormat.JPG),
        ], ui_state, "sample_image_format")

        self.components.button(top_frame, 0, 6, "sample now", self.sample_now)

        self.components.button(top_frame, 0, 7, "manual sample", self.open_manual_sample_window)

        self.components.label(sub_frame, 0, 0, "Non-EMA Sampling",
                         tooltip="Whether to include non-ema sampling when using ema.")
        self.components.switch(sub_frame, 0, 1, ui_state, "non_ema_sampling")

        self.components.label(sub_frame, 0, 2, "Samples to Tensorboard",
                         tooltip="Whether to include sample images in the Tensorboard output.")
        self.components.switch(sub_frame, 0, 3, ui_state, "samples_to_tensorboard")

    def build_backup_tab_content(self, frame, controller, ui_state):
        # backup after
        self.components.label(frame, 0, 0, "Backup After",
                         tooltip="The interval used when automatically creating model backups during training")
        self.components.time_entry(frame, 0, 1, ui_state, "backup_after", "backup_after_unit", max_width=5)

        # backup now
        self.components.button(frame, 0, 3, "backup now", self.backup_now, sticky="nw")

        # rolling backup
        self.components.label(frame, 1, 0, "Rolling Backup",
                         tooltip="If rolling backups are enabled, older backups are deleted automatically")
        self.components.switch(frame, 1, 1, ui_state, "rolling_backup")

        # rolling backup count
        self.components.label(frame, 2, 0, "Rolling Backup Count",
                         tooltip="Defines the number of backups to keep if rolling backups are enabled")
        self.components.entry(frame, 2, 1, ui_state, "rolling_backup_count", max_width=3)

        # backup before save
        self.components.label(frame, 3, 0, "Backup Before Save",
                         tooltip="Create a full backup before saving the final model")
        self.components.switch(frame, 3, 1, ui_state, "backup_before_save")

        # save after
        self.components.label(frame, 4, 0, "Save Every",
                         tooltip="The interval used when automatically saving the model during training")
        self.components.time_entry(frame, 4, 1, ui_state, "save_every", "save_every_unit", max_width=5)

        # save now
        self.components.button(frame, 4, 3, "save now", self.save_now, sticky="nw")

        # skip save
        self.components.label(frame, 5, 0, "Skip First",
                         tooltip="Start saving automatically after this interval has elapsed")
        self.components.entry(frame, 5, 1, ui_state, "save_skip_first", max_width=5, sticky="nw")

    def build_embedding_tab_content(self, frame, controller, ui_state):
        # embedding model name
        self.components.label(frame, 0, 0, "Base embedding",
                         tooltip="The base embedding to train on. Leave empty to create a new embedding")
        self.components.path_entry(
            frame, 0, 1, ui_state, "embedding.model_name",
            mode="file", path_modifier=path_util.json_path_modifier
        )

        # token count
        self.components.label(frame, 1, 0, "Token count",
                         tooltip="The token count used when creating a new embedding. Leave empty to auto detect from the initial embedding text.")
        self.components.entry(frame, 1, 1, ui_state, "embedding.token_count")

        # initial embedding text
        self.components.label(frame, 2, 0, "Initial embedding text",
                         tooltip="The initial embedding text used when creating a new embedding")
        self.components.entry(frame, 2, 1, ui_state, "embedding.initial_embedding_text")

        # embedding weight dtype
        self.components.label(frame, 3, 0, "Embedding Weight Data Type",
                         tooltip="The Embedding weight data type used for training. This can reduce memory consumption, but reduces precision")
        self.components.options_kv(frame, 3, 1, [
            ("float32", DataType.FLOAT_32),
            ("bfloat16", DataType.BFLOAT_16),
        ], ui_state, "embedding_weight_dtype")

        # placeholder
        self.components.label(frame, 4, 0, "Placeholder",
                         tooltip="The placeholder used when using the embedding in a prompt")
        self.components.entry(frame, 4, 1, ui_state, "embedding.placeholder")

        # output embedding
        self.components.label(frame, 5, 0, "Output embedding",
                         tooltip="Output embeddings are calculated at the output of the text encoder, not the input. This can improve results for larger text encoders and lower VRAM usage.")
        self.components.switch(frame, 5, 1, ui_state, "embedding.is_output_embedding")

    def build_tools_tab_content(self, frame, controller, ui_state):
        # dataset
        self.components.label(frame, 0, 0, "Dataset Tools",
                         tooltip="Open the captioning tool")
        self.components.button(frame, 0, 1, "Open", self.open_dataset_tool)

        # video tools
        self.components.label(frame, 1, 0, "Video Tools",
                         tooltip="Open the video tools")
        self.components.button(frame, 1, 1, "Open", self.open_video_tool)

        # convert model
        self.components.label(frame, 2, 0, "Convert Model Tools",
                         tooltip="Open the model conversion tool")
        self.components.button(frame, 2, 1, "Open", self.open_convert_model_tool)

        # sample
        self.components.label(frame, 3, 0, "Sampling Tool",
                         tooltip="Open the model sampling tool")
        self.components.button(frame, 3, 1, "Open", self.open_sampling_tool)

        self.components.label(frame, 4, 0, "Profiling Tool",
                         tooltip="Open the profiling tools.")
        self.components.button(frame, 4, 1, "Open", self.open_profiling_tool)


class TrainView(BaseTrainUIView, QMainWindow, metaclass=QtABCMeta):
    def __init__(self):
        QMainWindow.__init__(self)

        train_config = TrainConfig.default_values()
        ui_state = PySide6UIState(train_config)
        controller = TrainUIController(train_config)

        BaseTrainUIView.__init__(self, pyside6_components, controller, ui_state)
        self.controller.view = self

        self.setWindowTitle("OneTrainer")
        self.setWindowIcon(QIcon("resources/icons/icon.png"))
        self.setMinimumWidth(1150)
        self.resize(self.minimumWidth(), 715)

        self.status_label = None
        self.eta_label = None
        self.training_button = None
        self.export_button = None
        self.tabview: QTabWidget | None = None
        self._tab_widgets: dict[str, QWidget] = {}

        self.general_tab = None
        self.model_tab = None
        self.training_tab = None
        self.lora_tab = None
        self.cloud_tab = None
        self.concepts_tab = None
        self.sampling_tab = None
        self.additional_embeddings_tab = None

        central = QWidget(self)
        self.setCentralWidget(central)
        central_lo = QGridLayout(central)
        central_lo.setContentsMargins(0, 0, 0, 0)
        central_lo.setSpacing(0)
        central_lo.setRowStretch(1, 1)
        central_lo.setColumnStretch(0, 1)

        self.top_bar_component = self._build_top_bar(central)
        central_lo.addWidget(self.top_bar_component, 0, 0)

        self.tabview = QTabWidget(central)
        central_lo.addWidget(self.tabview, 1, 0)

        bottom = self._build_bottom_bar(central)
        central_lo.addWidget(bottom, 2, 0)

        self._create_tabs()
        self.change_training_method(self.controller.train_config.training_method)
        self._update_additional_embeddings_tab(self.controller.train_config.model_type)
        self.controller.sync_run_name(is_new_invocation=True)

        self._profiling_controller = ProfilingWindowController()
        self.profiling_window = self._profiling_controller.create_window(self, ProfilingWindowView)

        self.controller._check_start_always_on_tensorboard()
        self.workspace_dir_trace_id = self.ui_state.add_var_trace(
            "workspace_dir", self.controller._on_workspace_dir_change_trace
        )
        self.window_settings = MainWindowSettings(self, self.tabview, self._tab_widgets, QSettings())
        self.window_settings.restore()

    def closeEvent(self, event):
        if self.controller.training_thread is not None and self.controller.training_thread.is_alive():
            QMessageBox.warning(
                self,
                "Training in progress",
                "A training is currently running. Stop the training before closing the window.",
            )
            event.ignore()
            return
        self.top_bar_component.save_default()
        self.controller._stop_always_on_tensorboard()
        self.ui_state.remove_var_trace("workspace_dir", self.workspace_dir_trace_id)
        self.window_settings.save()
        event.accept()

    # --- BaseTrainUIView abstract method implementations ---

    def on_update_status(self, status: str):
        # Called from training thread — defer to main thread
        self.schedule_on_main_thread(lambda: self.status_label.setText(status))

    def on_training_started(self):
        self._set_training_button_style("running")

    def on_training_stopped(self, error_caught: bool):
        self.controller.sync_run_name(is_new_invocation=True)
        self.eta_label.setText("")
        self._set_training_button_style("idle")

    def on_training_stopping(self):
        self._set_training_button_style("stopping")

    def on_update_progress(self, epoch_step: int, max_step: int, epoch: int, max_epoch: int, eta_str: str | None):
        # Called from training thread — defer to main thread
        self.schedule_on_main_thread(lambda: self._do_update_progress(epoch_step, max_step, epoch, max_epoch, eta_str))

    def _do_update_progress(self, epoch_step: int, max_step: int, epoch: int, max_epoch: int, eta_str: str | None):
        self.set_step_progress(epoch_step, max_step)
        self.set_epoch_progress(epoch, max_epoch)
        self.eta_label.setText(f"ETA: {eta_str}" if eta_str is not None else "")

    def schedule_on_main_thread(self, fn: Callable):
        # The 3-argument form (msec, context, fn) is thread-safe: Qt marshals the call
        # to the thread where `self` lives (the main thread), unlike the 2-arg form.
        QTimer.singleShot(0, self, fn)

    def get_cloud_reattach(self) -> bool:
        return self.cloud_tab.reattach

    def save_default(self):
        self.top_bar_component.save_default()
        self.concepts_tab.save_current_config()
        self.sampling_tab.save_current_config()
        self.additional_embeddings_tab.save_current_config()

    def show_validation_errors(self, errors: list[str]):
        bullet_list = "\n".join(f"• {e}" for e in errors)
        QMessageBox.critical(self, "Cannot Start Training",
                             f"Please fix the following errors before training:\n\n{bullet_list}")

    def confirm(self, title: str, message: str) -> bool:
        buttons = QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel
        return QMessageBox.question(self, title, message, buttons) == QMessageBox.StandardButton.Ok

    def open_dataset_tool(self):
        self.wait_window(self.controller.open_dataset_tool(self, CaptionUIView))

    def open_video_tool(self):
        self.wait_window(self.controller.open_video_tool(self, VideoToolUIView))

    def open_convert_model_tool(self):
        self.wait_window(self.controller.open_convert_model_tool(self, ConvertModelUIView))

    def open_sampling_tool(self):
        self.controller.open_sampling_tool(self, SampleWindowView)

    def open_manual_sample_window(self):
        self.controller.open_manual_sample_window(self, SampleWindowView)

    def wait_window(self, window):
        window.exec()

    def show_window(self, window):
        window.show()

    def connect_window_closed(self, window, callback):
        window.finished.connect(lambda _: callback())

    # --- PySide6 layout builders ---

    def _build_top_bar(self, master):
        return TopBarView(
            master,
            TopBarController(self.controller.train_config),
            self.ui_state,
            self.change_model_type,
            self.change_training_method,
            self.load_preset,
        )

    def _build_bottom_bar(self, parent):
        frame = QWidget(parent)
        lo = QGridLayout(frame)
        lo.setColumnStretch(0, 1)
        lo.setColumnStretch(2, 2)
        lo.setColumnMinimumWidth(5, 10)

        status_frame = QWidget(frame)
        status_lo = QGridLayout(status_frame)
        status_lo.setContentsMargins(0, 0, 0, 0)
        lo.addWidget(status_frame, 0, 1)

        self.build_bottom_bar_content(frame, status_frame, self.controller, self.ui_state)
        self.training_button.setObjectName("trainingButton")
        self._set_training_button_style("idle")
        return frame

    def _create_scrollable_tab(self, configure_fn):
        tab_page = QWidget()
        tab_lo = pyside6_components._layout(tab_page)
        tab_lo.setRowStretch(0, 1)
        tab_lo.setColumnStretch(0, 1)
        scroll, frame = pyside6_components.scrollable_frame(tab_page)
        tab_lo.addWidget(scroll, 0, 0)
        configure_fn(frame)
        return tab_page

    def _configure_general_frame(self, frame):
        controller = GeneralTabController(
            local_gpu_count=self.controller._local_gpu_count,
            on_workspace_dir_change=self.controller._on_workspace_dir_change,
            on_always_on_tensorboard_toggle=self.controller._on_always_on_tensorboard_toggle,
        )
        self.general_tab = GeneralTabView(frame, controller, self.ui_state)
        lo = pyside6_components._layout(frame)
        lo.setColumnStretch(0, 1)
        lo.addWidget(self.general_tab, 0, 0)

    def _configure_data_frame(self, frame):
        lo = pyside6_components._layout(frame)
        lo.setColumnStretch(1, 1)
        lo.setColumnStretch(3, 1)
        self.build_data_tab_content(frame, self.controller, self.ui_state)
        pyside6_components._pack_form(frame)

    def _configure_backup_frame(self, frame):
        lo = pyside6_components._layout(frame)
        lo.setColumnStretch(4, 1)
        self.build_backup_tab_content(frame, self.controller, self.ui_state)
        fit_form(frame)

    def _configure_tools_frame(self, frame):
        from modules.ui.QueueTrainingUIView import QueueTrainingUIView
        pyside6_components.label(frame, 5, 0, "Training Queue", tooltip="Export Loop or Sweep training scripts")
        pyside6_components.button(frame, 5, 1, "Open", lambda: self.wait_window(
            self.controller.open_queue_tool(self, QueueTrainingUIView)))
        self.build_tools_tab_content(frame, self.controller, self.ui_state)
        pyside6_components._pack_form(frame)

    def _configure_embedding_frame(self, frame):
        self.build_embedding_tab_content(frame, self.controller, self.ui_state)
        pyside6_components._pack_form(frame)

    def _create_tabs(self):
        general_page = self._create_scrollable_tab(self._configure_general_frame)
        self.tabview.addTab(general_page, "general")
        self._tab_widgets["general"] = general_page

        self.model_tab = ModelTabView(None, ModelTabController(self.controller.train_config), self.ui_state)
        self.tabview.addTab(self.model_tab, "model")
        self._tab_widgets["model"] = self.model_tab

        data_page = self._create_scrollable_tab(self._configure_data_frame)
        self.tabview.addTab(data_page, "data")
        self._tab_widgets["data"] = data_page

        concepts_page = QWidget()
        self.concepts_tab = ConceptTabView(concepts_page, ConceptTabController(self.controller.train_config), self.ui_state)
        self.tabview.addTab(concepts_page, "concepts")
        self._tab_widgets["concepts"] = concepts_page

        self.training_tab = TrainingTabView(None, TrainingTabController(self.controller.train_config), self.ui_state)
        self.tabview.addTab(self.training_tab, "training")
        self._tab_widgets["training"] = self.training_tab

        sampling_page = self.create_sampling_tab()
        self.tabview.addTab(sampling_page, "sampling")
        self._tab_widgets["sampling"] = sampling_page

        backup_page = self._create_scrollable_tab(self._configure_backup_frame)
        self.tabview.addTab(backup_page, "backup")
        self._tab_widgets["backup"] = backup_page

        tools_page = self._create_scrollable_tab(self._configure_tools_frame)
        self.tabview.addTab(tools_page, "tools")
        self._tab_widgets["tools"] = tools_page

        additional_embeddings_page = QWidget()
        self.additional_embeddings_tab = AdditionalEmbeddingsTabView(
            additional_embeddings_page,
            AdditionalEmbeddingsTabController(self.controller.train_config),
            self.ui_state,
        )
        self.tabview.addTab(additional_embeddings_page, "additional embeddings")
        self._tab_widgets["additional embeddings"] = additional_embeddings_page

        self.cloud_tab = CloudTabView(None, CloudTabController(self.controller.train_config, self), self.ui_state)
        self.tabview.addTab(self.cloud_tab, "cloud")
        self._tab_widgets["cloud"] = self.cloud_tab

    def create_sampling_tab(self):
        tab_page = QWidget()
        tab_lo = QGridLayout(tab_page)
        tab_lo.setContentsMargins(0, 0, 0, 0)
        tab_lo.setSpacing(0)
        tab_lo.setRowStretch(0, 0)
        tab_lo.setRowStretch(1, 1)
        tab_lo.setColumnStretch(0, 1)

        top_frame = QWidget(tab_page)
        tab_lo.addWidget(top_frame, 0, 0)
        top_lo = pyside6_components._layout(top_frame)
        top_lo.setContentsMargins(pyside6_components.PAD, pyside6_components.PAD, pyside6_components.PAD, pyside6_components.PAD)
        top_lo.setColumnStretch(8, 1)

        sub_frame = QWidget(top_frame)
        pyside6_components._layout(top_frame).addWidget(sub_frame, 1, 0, 1, 8)

        self.build_sampling_tab_header(top_frame, sub_frame, self.controller, self.ui_state)
        pyside6_components._layout(sub_frame).setColumnStretch(4, 1)

        sampling_container = QWidget(tab_page)
        tab_lo.addWidget(sampling_container, 1, 0)
        self.sampling_tab = SamplingTabView(
            sampling_container, SamplingTabController(self.controller.train_config), self.ui_state
        )

        return tab_page

    def open_profiling_tool(self):
        self.profiling_window.show()

    def change_model_type(self, model_type: ModelType):
        if self.model_tab:
            self.model_tab.refresh_ui()
        if self.training_tab:
            self.training_tab.refresh_ui()
        if self.lora_tab:
            self.lora_tab.refresh_ui()
        if self.sampling_tab:
            self.sampling_tab.refresh_ui()
        self._update_additional_embeddings_tab(model_type)

    def _update_additional_embeddings_tab(self, model_type: ModelType):
        # additional embeddings only apply to models that support embedding training
        supported = TrainingMethod.EMBEDDING in model_type.supported_training_methods()
        page = self._tab_widgets.get("additional embeddings")
        if page is not None:
            self.tabview.setTabVisible(self.tabview.indexOf(page), supported)

    def change_training_method(self, training_method: TrainingMethod):
        if not self.tabview:
            return

        if self.model_tab:
            self.model_tab.refresh_ui()

        if training_method != TrainingMethod.LORA and 'LoRA' in self._tab_widgets:
            self.tabview.removeTab(self.tabview.indexOf(self._tab_widgets['LoRA']))
            del self._tab_widgets['LoRA']
            self.lora_tab = None
        if training_method != TrainingMethod.EMBEDDING and 'embedding' in self._tab_widgets:
            self.tabview.removeTab(self.tabview.indexOf(self._tab_widgets['embedding']))
            del self._tab_widgets['embedding']

        if training_method == TrainingMethod.LORA and 'LoRA' not in self._tab_widgets:
            self.lora_tab = LoraTabView(None, LoraTabController(self.controller.train_config), self.ui_state)
            self.tabview.addTab(self.lora_tab, 'LoRA')
            self._tab_widgets['LoRA'] = self.lora_tab
        if training_method == TrainingMethod.EMBEDDING and 'embedding' not in self._tab_widgets:
            tab_page = self._create_scrollable_tab(self._configure_embedding_frame)
            self.tabview.addTab(tab_page, 'embedding')
            self._tab_widgets['embedding'] = tab_page

    def load_preset(self):
        if self.general_tab:
            self.general_tab.refresh_availability()
        if self.additional_embeddings_tab:
            self.additional_embeddings_tab.refresh_ui()

    def _set_training_button_style(self, mode: str):
        if not self.training_button:
            return
        styles = {
            "idle":     ("Start Training", True,  "confirm-action"),
            "running":  ("Stop Training",  True,  "destructive-action"),
            "stopping": ("Stopping...",    False, "destructive-action"),
        }
        text, enabled, action_class = styles.get(mode, styles["idle"])
        button = self.training_button
        button.setText(text)
        button.setEnabled(enabled)
        button.setProperty("class", action_class)
        button.style().unpolish(button)
        button.style().polish(button)
        button.update()

    def export_training(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Training Config", "config.json",
            "JSON Files (*.json);;All Files (*.*)"
        )
        if file_path:
            self.controller.export_training(file_path)

    def generate_debug_package(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Directory to Save Debug Package", ".")
        if not dir_path:
            return
        self.controller.generate_debug_package(Path(dir_path) / "OneTrainer_debug_report.zip")
