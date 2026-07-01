from collections.abc import Callable
from pathlib import Path

from modules.ui.AdditionalEmbeddingsTabController import AdditionalEmbeddingsTabController
from modules.ui.BaseTrainUIView import BaseTrainUIView
from modules.ui.CloudTabController import CloudTabController
from modules.ui.ConceptTabController import ConceptTabController
from modules.ui.LoraTabController import LoraTabController
from modules.ui.ModelTabController import ModelTabController
from modules.ui.ProfilingWindowController import ProfilingWindowController
from modules.ui.PySide6AdditionalEmbeddingsTabView import PySide6AdditionalEmbeddingsTabView
from modules.ui.PySide6CaptionUIView import PySide6CaptionUIView
from modules.ui.PySide6CloudTabView import PySide6CloudTabView
from modules.ui.PySide6ConceptTabView import PySide6ConceptTabView
from modules.ui.PySide6ConvertModelUIView import PySide6ConvertModelUIView
from modules.ui.PySide6LoraTabView import PySide6LoraTabView
from modules.ui.PySide6ModelTabView import PySide6ModelTabView
from modules.ui.PySide6ProfilingWindowView import PySide6ProfilingWindowView
from modules.ui.PySide6SampleWindowView import PySide6SampleWindowView
from modules.ui.PySide6SamplingTabView import PySide6SamplingTabView
from modules.ui.PySide6TopBarView import PySide6TopBarView
from modules.ui.PySide6TrainingTabView import PySide6TrainingTabView
from modules.ui.PySide6VideoToolUIView import PySide6VideoToolUIView
from modules.ui.SamplingTabController import SamplingTabController
from modules.ui.TopBarController import TopBarController
from modules.ui.TrainingTabController import TrainingTabController
from modules.ui.TrainUIController import TrainUIController
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.ui import pyside6_components
from modules.util.ui.pyside6_util import QtABCMeta
from modules.util.ui.PySide6UIState import PySide6UIState

from PySide6.QtCore import QTimer
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QFileDialog, QGridLayout, QMainWindow, QMessageBox, QTabWidget, QWidget


class PySide6TrainView(BaseTrainUIView, QMainWindow, metaclass=QtABCMeta):
    def __init__(self):
        QMainWindow.__init__(self)

        train_config = TrainConfig.default_values()
        ui_state = PySide6UIState(train_config)
        controller = TrainUIController(train_config)

        BaseTrainUIView.__init__(self, pyside6_components, controller, ui_state)
        self.controller.view = self

        self.setWindowTitle("OneTrainer")
        self.setWindowIcon(QIcon("resources/icons/icon.png"))
        self.resize(1100, 740)

        self.status_label = None
        self.eta_label = None
        self.training_button = None
        self.export_button = None
        self.tabview: QTabWidget | None = None
        self._tab_widgets: dict[str, QWidget] = {}

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

        self._profiling_controller = ProfilingWindowController()
        self.profiling_window = self._profiling_controller.create_window(self, PySide6ProfilingWindowView)

        self.controller._check_start_always_on_tensorboard()
        self.workspace_dir_trace_id = self.ui_state.add_var_trace(
            "workspace_dir", self.controller._on_workspace_dir_change_trace
        )

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
        event.accept()

    # --- BaseTrainUIView abstract method implementations ---

    def on_update_status(self, status: str):
        # Called from training thread — defer to main thread
        self.schedule_on_main_thread(lambda: self.status_label.setText(status))

    def on_training_started(self):
        self._set_training_button_style("running")

    def on_training_stopped(self, error_caught: bool):
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

    def open_dataset_tool(self):
        self.wait_window(self.controller.open_dataset_tool(self, PySide6CaptionUIView))

    def open_video_tool(self):
        self.wait_window(self.controller.open_video_tool(self, PySide6VideoToolUIView))

    def open_convert_model_tool(self):
        self.wait_window(self.controller.open_convert_model_tool(self, PySide6ConvertModelUIView))

    def open_sampling_tool(self):
        self.controller.open_sampling_tool(self, PySide6SampleWindowView)

    def open_manual_sample_window(self):
        self.controller.open_manual_sample_window(self, PySide6SampleWindowView)

    def wait_window(self, window):
        window.exec()

    def show_window(self, window):
        window.show()

    def connect_window_closed(self, window, callback):
        window.finished.connect(lambda _: callback())

    # --- PySide6 layout builders ---

    def _build_top_bar(self, master):
        return PySide6TopBarView(
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

        status_frame = QWidget(frame)
        status_lo = QGridLayout(status_frame)
        status_lo.setContentsMargins(0, 0, 0, 0)
        lo.addWidget(status_frame, 0, 1)

        self.build_bottom_bar_content(frame, status_frame, self.controller, self.ui_state)
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
        lo = pyside6_components._layout(frame)
        lo.setColumnStretch(1, 1)
        lo.setColumnStretch(3, 1)
        self.build_general_tab_content(frame, self.controller, self.ui_state)
        pyside6_components._pack_form(frame)

    def _configure_data_frame(self, frame):
        lo = pyside6_components._layout(frame)
        lo.setColumnStretch(1, 1)
        lo.setColumnStretch(3, 1)
        self.build_data_tab_content(frame, self.controller, self.ui_state)
        pyside6_components._pack_form(frame)

    def _configure_backup_frame(self, frame):
        lo = pyside6_components._layout(frame)
        lo.setColumnStretch(1, 1)
        lo.setColumnStretch(3, 1)
        self.build_backup_tab_content(frame, self.controller, self.ui_state)
        pyside6_components._pack_form(frame)

    def _configure_tools_frame(self, frame):
        self.build_tools_tab_content(frame, self.controller, self.ui_state)
        pyside6_components._pack_form(frame)

    def _configure_embedding_frame(self, frame):
        self.build_embedding_tab_content(frame, self.controller, self.ui_state)
        pyside6_components._pack_form(frame)

    def _create_tabs(self):
        general_page = self._create_scrollable_tab(self._configure_general_frame)
        self.tabview.addTab(general_page, "general")
        self._tab_widgets["general"] = general_page

        self.model_tab = PySide6ModelTabView(None, ModelTabController(self.controller.train_config), self.ui_state)
        self.tabview.addTab(self.model_tab, "model")
        self._tab_widgets["model"] = self.model_tab

        data_page = self._create_scrollable_tab(self._configure_data_frame)
        self.tabview.addTab(data_page, "data")
        self._tab_widgets["data"] = data_page

        concepts_page = QWidget()
        self.concepts_tab = PySide6ConceptTabView(concepts_page, ConceptTabController(self.controller.train_config), self.ui_state)
        self.tabview.addTab(concepts_page, "concepts")
        self._tab_widgets["concepts"] = concepts_page

        self.training_tab = PySide6TrainingTabView(None, TrainingTabController(self.controller.train_config), self.ui_state)
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
        self.additional_embeddings_tab = PySide6AdditionalEmbeddingsTabView(
            additional_embeddings_page,
            AdditionalEmbeddingsTabController(self.controller.train_config),
            self.ui_state,
        )
        self.tabview.addTab(additional_embeddings_page, "additional embeddings")
        self._tab_widgets["additional embeddings"] = additional_embeddings_page

        self.cloud_tab = PySide6CloudTabView(None, CloudTabController(self.controller.train_config, self), self.ui_state)
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
        self.sampling_tab = PySide6SamplingTabView(
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
            self.lora_tab = PySide6LoraTabView(None, LoraTabController(self.controller.train_config), self.ui_state)
            self.tabview.addTab(self.lora_tab, 'LoRA')
            self._tab_widgets['LoRA'] = self.lora_tab
        if training_method == TrainingMethod.EMBEDDING and 'embedding' not in self._tab_widgets:
            tab_page = self._create_scrollable_tab(self._configure_embedding_frame)
            self.tabview.addTab(tab_page, 'embedding')
            self._tab_widgets['embedding'] = tab_page

    def load_preset(self):
        if self.additional_embeddings_tab:
            self.additional_embeddings_tab.refresh_ui()

    def _set_training_button_style(self, mode: str):
        if not self.training_button:
            return
        styles = {
            "idle":     ("Start Training", True,  "#198754", "white"),
            "running":  ("Stop Training",  True,  "#dc3545", "white"),
            "stopping": ("Stopping...",    False, "#dc3545", "white"),
        }
        text, enabled, bg, fg = styles.get(mode, ("Start Training", True, "#198754", "white"))
        self.training_button.setText(text)
        self.training_button.setEnabled(enabled)
        self.training_button.setStyleSheet(
            f"QPushButton {{ background-color: {bg}; color: {fg}; }}"
            f"QPushButton:disabled {{ background-color: {bg}; color: {fg}; }}"
        )

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
