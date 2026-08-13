import threading

from modules.modelSampler.BaseModelSampler import (
    ModelSamplerOutput,
)
from modules.ui.BaseSampleWindowView import BaseSampleWindowView
from modules.ui.PySide6SampleFrameView import PySide6SampleFrameView
from modules.ui.SampleFrameController import SampleFrameController
from modules.ui.SampleWindowController import SampleWindowController
from modules.util.enum.FileType import FileType
from modules.util.ui import pyside6_components
from modules.util.ui.PySide6UIState import PySide6UIState

from PIL.ImageQt import ImageQt
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QDialog, QGridLayout, QHBoxLayout, QLabel, QProgressBar, QPushButton, QWidget


class PySide6SampleWindowView(BaseSampleWindowView, QDialog):
    def __init__(self, parent, controller: SampleWindowController):
        QDialog.__init__(self, parent)
        BaseSampleWindowView.__init__(self, pyside6_components)

        self.setWindowTitle("Sample")
        self.resize(1200, 800)

        self.ui_state = PySide6UIState(controller.sample)

        if controller.use_external_model:
            controller.callbacks.set_on_sample_custom(self._update_preview)
            controller.callbacks.set_on_update_sample_custom_progress(self._update_progress)

        outer = QGridLayout(self)
        outer.setRowStretch(1, 1)
        outer.setColumnStretch(1, 1)

        model_type = controller.get_model_type()
        frame_controller = SampleFrameController(controller.sample, model_type)

        prompt_frame = PySide6SampleFrameView(self, frame_controller, self.ui_state, include_settings=False)
        outer.addWidget(prompt_frame, 0, 0, 1, 2)

        settings_frame = PySide6SampleFrameView(self, frame_controller, self.ui_state, include_prompt=False)
        outer.addWidget(settings_frame, 1, 0)

        self._image_label = QLabel(self)
        self._image_label.setFixedSize(512, 512)
        self._image_label.setAlignment(Qt.AlignCenter)
        self._image_label.setStyleSheet("background: black;")
        outer.addWidget(self._image_label, 1, 1, 2, 1)

        # gallery navigation, on the same row as the sample button
        self._nav_widget = QWidget(self)
        nav_layout = QHBoxLayout(self._nav_widget)
        self._prev_button = QPushButton("◀", self._nav_widget)
        self._prev_button.clicked.connect(lambda: self._step_gallery(-1))
        self._counter_label = QLabel("", self._nav_widget)
        self._next_button = QPushButton("▶", self._nav_widget)
        self._next_button.clicked.connect(lambda: self._step_gallery(1))
        nav_layout.addWidget(self._prev_button)
        nav_layout.addWidget(self._counter_label)
        nav_layout.addWidget(self._next_button)
        outer.addWidget(self._nav_widget, 3, 1, alignment=Qt.AlignCenter)

        self._progress = QProgressBar(self)
        self._progress.setRange(0, 1000)
        outer.addWidget(self._progress, 2, 0)

        sample_btn = QPushButton("sample", self)
        def _on_sample():
            # With an external model (manual sample during training) do_sample
            # only enqueues a command for the training thread to execute, so it
            # returns immediately; queueing several is fine, no thread needed.
            if controller.use_external_model:
                controller.do_sample(self._update_preview, self._update_progress)
                return

            # Standalone sampling runs the model synchronously, so run it in a
            # background thread to keep the Qt event loop responsive, and disable
            # the button while a sample is in flight to avoid concurrent runs.
            sample_btn.setEnabled(False)

            def _run():
                try:
                    controller.do_sample(self._update_preview, self._update_progress)
                finally:
                    self.schedule_on_main_thread(lambda: sample_btn.setEnabled(True))

            threading.Thread(target=_run, daemon=True).start()
        sample_btn.clicked.connect(_on_sample)
        outer.addWidget(sample_btn, 3, 0)

        self._render_gallery()


    def schedule_on_main_thread(self, fn):
        QTimer.singleShot(0, self, fn)

    def _update_preview(self, sampler_output: ModelSamplerOutput):
        # Called from training thread — capture data and dispatch to main thread
        if sampler_output.file_type == FileType.IMAGE:
            image = sampler_output.data
            self.schedule_on_main_thread(lambda: self._do_update_preview(image))

    def _do_update_preview(self, image):
        # gallery mutation runs on the main thread, so state stays consistent
        self.gallery_add(image)
        self._render_gallery()

    def _step_gallery(self, delta):
        self.gallery_step(delta)
        self._render_gallery()

    def _render_gallery(self):
        image = self.gallery_current
        if image is not None:
            pixmap = QPixmap.fromImage(ImageQt(image.convert("RGBA")))
            self._image_label.setFixedSize(pixmap.size())
            self._image_label.setPixmap(pixmap)

        self._counter_label.setText(f"{self.gallery_index + 1} / {self.gallery_count}")
        self._prev_button.setEnabled(self.gallery_index > 0)
        self._next_button.setEnabled(self.gallery_index < self.gallery_count - 1)

    def _update_progress(self, progress: int, max_progress: int):
        # Called from training thread — dispatch to main thread
        value = int(progress / max_progress * 1000)
        self.schedule_on_main_thread(lambda: self._progress.setValue(value))
