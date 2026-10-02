import os
import subprocess
import sys
import traceback
from collections.abc import Callable

from modules.ui.SimpleTrainController import (
    DEFAULT_MODEL_GROUP,
    PresetInfo,
    SimpleTrainController,
    SimpleTrainSettings,
)
from modules.ui.TrainUIController import TrainUIController
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.i18n import t
from modules.util.ui.pyside6_components import NoScrollComboBox, NoScrollDoubleSpinBox, NoScrollSpinBox

from PySide6.QtCore import QTimer, QUrl
from PySide6.QtGui import QDesktopServices, QIcon
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

CAPTION_SOURCES = [
    ("Text file next to each image (.txt)", "sample"),
    ("Image file name", "filename"),
]

RESOLUTIONS = ["512", "768", "1024"]


class PySide6SimpleTrainView(QMainWindow):
    """A single-page training UI: pick a preset, a folder of images and an output name, then train."""

    def __init__(self):
        super().__init__()

        self.train_config = TrainConfig.default_values()
        self.controller = SimpleTrainController(self.train_config)
        self.train_controller = TrainUIController(self.train_config)
        self.train_controller.view = self

        self.settings = SimpleTrainSettings.load()
        self.preset_groups = self.controller.load_preset_groups()
        self._loading = False
        self._image_count = 0

        self.setWindowTitle(t("OneTrainer - Simple Training"))
        self.setWindowIcon(QIcon("resources/icons/icon.png"))
        self.resize(720, 820)

        central = QWidget(self)
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        scroll = QScrollArea(central)
        scroll.setWidgetResizable(True)
        form_host = QWidget()
        scroll.setWidget(form_host)
        self.form_layout = QVBoxLayout(form_host)
        root.addWidget(scroll, 1)

        self._build_model_group()
        self._build_data_group()
        self._build_output_group()
        self._build_training_group()
        self._build_sample_group()
        self.form_layout.addStretch(1)

        root.addWidget(self._build_bottom_bar())

        for spin in (self.epochs_spin, self.repeats_spin, self.batch_spin):
            spin.valueChanged.connect(self._update_steps_info)

        self._apply_settings_to_ui()

    # --- layout ---

    @staticmethod
    def _path_row(line_edit: QLineEdit, browse: Callable[[], None]) -> QWidget:
        row = QWidget()
        lo = QHBoxLayout(row)
        lo.setContentsMargins(0, 0, 0, 0)
        lo.addWidget(line_edit, 1)
        button = QPushButton("...")
        button.setFixedWidth(32)
        button.clicked.connect(browse)
        lo.addWidget(button)
        return row

    def _group(self, title: str) -> QFormLayout:
        box = QGroupBox(t(title))
        form = QFormLayout(box)
        self.form_layout.addWidget(box)
        return form

    @staticmethod
    def _hint(text: str) -> QLabel:
        label = QLabel(t(text))
        label.setWordWrap(True)
        label.setStyleSheet("color: #666666;")
        return label

    def _build_model_group(self):
        form = self._group("1. Model")

        self.model_combo = NoScrollComboBox()
        self.model_combo.addItems(list(self.preset_groups.keys()))
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        form.addRow(t("Model:"), self.model_combo)

        self.preset_combo = NoScrollComboBox()
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        form.addRow(t("Training type:"), self.preset_combo)
        form.addRow("", self._hint(
            "LoRA trains a small add-on file and needs much less VRAM. "
            "The number in the name (e.g. 16GB) is the VRAM the preset is tuned for."
        ))

        self.base_model_edit = QLineEdit()
        self.base_model_edit.setToolTip(t("A Hugging Face repository name or a local Diffusers model folder."))
        form.addRow(t("Base model:"), self._path_row(self.base_model_edit, self._browse_base_model))

        self.transformer_model_edit = QLineEdit()
        self.transformer_model_edit.setPlaceholderText(t("Leave empty to train the base model itself"))
        self.transformer_model_label = QLabel(t("Custom model file:"))
        self.transformer_model_row = self._path_row(self.transformer_model_edit, self._browse_transformer_model)
        form.addRow(self.transformer_model_label, self.transformer_model_row)
        self.transformer_model_hint = self._hint(
            "Optional. A single .safetensors or .gguf file, e.g. a finetuned model from ComfyUI's "
            "models/diffusion_models folder. Keep the official base model above: it still provides "
            "the text encoder and VAE."
        )
        form.addRow("", self.transformer_model_hint)

    def _build_data_group(self):
        form = self._group("2. Training images")

        self.dataset_edit = QLineEdit()
        self.dataset_edit.editingFinished.connect(self._update_dataset_info)
        form.addRow(t("Image folder:"), self._path_row(self.dataset_edit, self._browse_dataset))

        self.subdirs_check = QCheckBox(t("Include subfolders"))
        self.subdirs_check.toggled.connect(self._update_dataset_info)
        form.addRow("", self.subdirs_check)

        self.caption_combo = NoScrollComboBox()
        for label, _ in CAPTION_SOURCES:
            self.caption_combo.addItem(t(label))
        self.caption_combo.currentIndexChanged.connect(self._update_dataset_info)
        form.addRow(t("Captions from:"), self.caption_combo)

        self.dataset_info_label = QLabel()
        self.dataset_info_label.setWordWrap(True)
        form.addRow("", self.dataset_info_label)

        self.repeats_spin = NoScrollDoubleSpinBox()
        self.repeats_spin.setRange(0.1, 1000)
        self.repeats_spin.setDecimals(1)
        self.repeats_spin.setToolTip(t("How many times each image is shown per epoch."))
        form.addRow(t("Repeats per image:"), self.repeats_spin)

    def _build_output_group(self):
        form = self._group("3. Output")

        self.output_name_edit = QLineEdit()
        self.output_name_edit.textChanged.connect(self._update_output_info)
        form.addRow(t("Name:"), self.output_name_edit)

        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.textChanged.connect(self._update_output_info)
        form.addRow(t("Output folder:"), self._path_row(self.output_dir_edit, self._browse_output_dir))

        self.output_info_label = self._hint("")
        form.addRow("", self.output_info_label)

    def _build_training_group(self):
        form = self._group("4. Training settings")

        self.epochs_spin = NoScrollSpinBox()
        self.epochs_spin.setRange(1, 100000)
        self.epochs_spin.setToolTip(t("How many times the whole image folder is trained on."))
        form.addRow(t("Epochs:"), self.epochs_spin)

        self.lr_edit = QLineEdit()
        self.lr_edit.setToolTip(t("Usually keep the preset value. Lower it if the result looks overcooked."))
        form.addRow(t("Learning rate:"), self.lr_edit)

        self.rank_spin = NoScrollSpinBox()
        self.rank_spin.setRange(1, 1024)
        self.rank_spin.setToolTip(t("Higher rank learns more detail but makes a bigger file. 16-32 is common."))
        self.rank_label = QLabel(t("LoRA rank:"))
        form.addRow(self.rank_label, self.rank_spin)

        self.alpha_spin = NoScrollDoubleSpinBox()
        self.alpha_spin.setRange(0.01, 1024)
        self.alpha_spin.setDecimals(2)
        self.alpha_label = QLabel(t("LoRA alpha:"))
        form.addRow(self.alpha_label, self.alpha_spin)

        self.resolution_combo = NoScrollComboBox()
        self.resolution_combo.setEditable(True)
        self.resolution_combo.addItems(RESOLUTIONS)
        self.resolution_combo.setToolTip(t("Images are resized to about this size. Higher needs more VRAM."))
        form.addRow(t("Resolution:"), self.resolution_combo)

        self.batch_spin = NoScrollSpinBox()
        self.batch_spin.setRange(1, 256)
        form.addRow(t("Batch size:"), self.batch_spin)

        self.steps_label = self._hint("")
        form.addRow("", self.steps_label)

        self.save_every_spin = NoScrollSpinBox()
        self.save_every_spin.setRange(0, 100000)
        self.save_every_spin.setSpecialValueText(t("Only at the end"))
        self.save_every_spin.setToolTip(t("Also save an intermediate copy every N epochs, so you can pick the best one."))
        form.addRow(t("Save every N epochs:"), self.save_every_spin)

        reset_button = QPushButton(t("Reset training settings to preset"))
        reset_button.clicked.connect(self._reset_to_preset)
        form.addRow("", reset_button)

    def _build_sample_group(self):
        form = self._group("5. Preview images (optional)")

        self.sample_prompt_edit = QLineEdit()
        self.sample_prompt_edit.setPlaceholderText(t("Leave empty to skip previews"))
        form.addRow(t("Preview prompt:"), self.sample_prompt_edit)

        self.sample_every_spin = NoScrollSpinBox()
        self.sample_every_spin.setRange(1, 100000)
        form.addRow(t("Preview every N epochs:"), self.sample_every_spin)
        form.addRow("", self._hint("Preview images are saved in workspace/<name>/samples."))

    def _build_bottom_bar(self) -> QWidget:
        bar = QWidget()
        lo = QVBoxLayout(bar)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        lo.addWidget(self.progress_bar)

        status_row = QHBoxLayout()
        self.status_label = QLabel(t("Ready"))
        self.eta_label = QLabel("")
        status_row.addWidget(self.status_label, 1)
        status_row.addWidget(self.eta_label)
        lo.addLayout(status_row)

        buttons = QHBoxLayout()
        advanced_button = QPushButton(t("Advanced mode..."))
        advanced_button.setToolTip(t("Open these settings in the full OneTrainer UI."))
        advanced_button.clicked.connect(self._open_advanced)
        buttons.addWidget(advanced_button)

        self.restore_defaults_button = QPushButton(t("Restore defaults"))
        self.restore_defaults_button.setToolTip(t("Reset every setting on this page to its default value."))
        self.restore_defaults_button.clicked.connect(self._restore_all_defaults)
        buttons.addWidget(self.restore_defaults_button)

        open_output_button = QPushButton(t("Open output folder"))
        open_output_button.clicked.connect(self._open_output_dir)
        buttons.addWidget(open_output_button)
        buttons.addStretch(1)

        self.training_button = QPushButton()
        self.training_button.setMinimumWidth(160)
        self.training_button.clicked.connect(self._on_training_button)
        buttons.addWidget(self.training_button)
        lo.addLayout(buttons)

        self._set_training_button_style("idle")
        return bar

    # --- settings <-> ui ---

    def _current_preset(self) -> PresetInfo | None:
        presets = self.preset_groups.get(self.model_combo.currentText(), [])
        index = self.preset_combo.currentIndex()
        return presets[index] if 0 <= index < len(presets) else None

    def _apply_settings_to_ui(self):
        s = self.settings
        self._loading = True
        try:
            group, index = self._find_preset(s.preset_path)
            self.model_combo.setCurrentText(group)
            self._fill_preset_combo(group)
            self.preset_combo.setCurrentIndex(index)

            self.base_model_edit.setText(s.base_model_name)
            self.transformer_model_edit.setText(s.transformer_model_name)
            self.dataset_edit.setText(s.dataset_path)
            self.subdirs_check.setChecked(s.include_subdirectories)
            sources = [value for _, value in CAPTION_SOURCES]
            self.caption_combo.setCurrentIndex(sources.index(s.caption_source) if s.caption_source in sources else 0)
            self.repeats_spin.setValue(s.repeats)
            self.output_name_edit.setText(s.output_name)
            self.output_dir_edit.setText(s.output_dir)
            self.epochs_spin.setValue(s.epochs)
            self.lr_edit.setText(f"{s.learning_rate:g}")
            self.rank_spin.setValue(s.lora_rank)
            self.alpha_spin.setValue(s.lora_alpha)
            self.resolution_combo.setCurrentText(s.resolution)
            self.batch_spin.setValue(s.batch_size)
            self.save_every_spin.setValue(s.save_every_epochs)
            self.sample_prompt_edit.setText(s.sample_prompt)
            self.sample_every_spin.setValue(s.sample_every_epochs)
        finally:
            self._loading = False

        if not s.preset_path:
            # first start: take everything from the default preset
            self._reset_to_preset(include_base_model=True)

        preset = self._current_preset()
        if preset is not None and preset.supports_transformer_override and os.path.isfile(s.base_model_name):
            # a single-file checkpoint can't be a base model: use it as the custom model file instead
            self.transformer_model_edit.setText(s.base_model_name)
            self.base_model_edit.setText(preset.base_model_name)
        self._update_method_widgets()
        self._update_dataset_info()
        self._update_output_info()

    def _find_preset(self, path: str) -> tuple[str, int]:
        for group, presets in self.preset_groups.items():
            for i, preset in enumerate(presets):
                if os.path.normcase(os.path.abspath(preset.path)) == os.path.normcase(os.path.abspath(path or "")):
                    return group, i
        group = DEFAULT_MODEL_GROUP if DEFAULT_MODEL_GROUP in self.preset_groups else next(iter(self.preset_groups), "")
        return group, 0

    def _collect_settings(self) -> SimpleTrainSettings | None:
        preset = self._current_preset()
        try:
            learning_rate = float(self.lr_edit.text().strip())
        except ValueError:
            QMessageBox.critical(self, t("Cannot Start Training"), t("The learning rate is not a valid number."))
            return None

        return SimpleTrainSettings(
            preset_path=preset.path if preset else "",
            base_model_name=self.base_model_edit.text().strip(),
            transformer_model_name=self.transformer_model_edit.text().strip()
            if preset is not None and preset.supports_transformer_override else "",
            dataset_path=self.dataset_edit.text().strip(),
            include_subdirectories=self.subdirs_check.isChecked(),
            caption_source=CAPTION_SOURCES[self.caption_combo.currentIndex()][1],
            output_name=self.output_name_edit.text().strip(),
            output_dir=self.output_dir_edit.text().strip(),
            epochs=self.epochs_spin.value(),
            repeats=self.repeats_spin.value(),
            learning_rate=learning_rate,
            lora_rank=self.rank_spin.value(),
            lora_alpha=self.alpha_spin.value(),
            resolution=self.resolution_combo.currentText().strip(),
            batch_size=self.batch_spin.value(),
            save_every_epochs=self.save_every_spin.value(),
            sample_prompt=self.sample_prompt_edit.text().strip(),
            sample_every_epochs=self.sample_every_spin.value(),
        )

    def _save_settings(self):
        settings = self._collect_settings()
        if settings is not None:
            self.settings = settings
            settings.save()

    # --- events ---

    def _fill_preset_combo(self, group: str):
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        for preset in self.preset_groups.get(group, []):
            self.preset_combo.addItem(preset.name)
        self.preset_combo.blockSignals(False)

    def _on_model_changed(self, group: str):
        if self._loading:
            return
        self._fill_preset_combo(group)
        self.preset_combo.setCurrentIndex(0)
        self.transformer_model_edit.clear()  # a checkpoint of another model can't be used
        self._on_preset_changed()

    def _on_preset_changed(self, *_):
        if self._loading:
            return
        self._reset_to_preset(include_base_model=True)
        self._update_method_widgets()
        self._update_output_info()

    def _reset_to_preset(self, include_base_model: bool = False):
        preset = self._current_preset()
        if preset is None:
            return
        if include_base_model:
            self.base_model_edit.setText(preset.base_model_name)
        self.epochs_spin.setValue(preset.epochs)
        self.lr_edit.setText(f"{preset.learning_rate:g}")
        self.rank_spin.setValue(preset.lora_rank)
        self.alpha_spin.setValue(preset.lora_alpha)
        self.resolution_combo.setCurrentText(preset.resolution)
        self.batch_spin.setValue(preset.batch_size)

    def _restore_all_defaults(self):
        if self._is_training():
            return
        answer = QMessageBox.question(
            self, t("Restore defaults"),
            t("Reset every setting on this page (model, folders, training settings and preview) to its default value?"),
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.settings = SimpleTrainSettings()
        self._apply_settings_to_ui()
        self._save_settings()

    def _update_method_widgets(self):
        preset = self._current_preset()
        is_lora = preset is not None and preset.training_method == TrainingMethod.LORA
        for widget in (self.rank_label, self.rank_spin, self.alpha_label, self.alpha_spin):
            widget.setVisible(is_lora)
        supports_override = preset is not None and preset.supports_transformer_override
        for widget in (self.transformer_model_label, self.transformer_model_row, self.transformer_model_hint):
            widget.setVisible(supports_override)

    def _update_dataset_info(self, *_):
        info = self.controller.scan_dataset(self.dataset_edit.text().strip(), self.subdirs_check.isChecked())
        self._image_count = info.image_count
        self._update_steps_info()
        path = self.dataset_edit.text().strip()
        if not path:
            text, color = t("Choose the folder with your training images."), "#666666"
        elif not os.path.isdir(path):
            text, color = t("Folder not found."), "#dc3545"
        elif info.image_count == 0:
            text, color = t("No images found in this folder."), "#dc3545"
        else:
            uses_txt = CAPTION_SOURCES[self.caption_combo.currentIndex()][1] == "sample"
            if uses_txt and info.caption_count < info.image_count:
                text = t("Found {images} images, {missing} of them have no .txt caption.").format(
                    images=info.image_count, missing=info.image_count - info.caption_count)
                color = "#b8860b"
            elif not uses_txt and info.caption_count > 0:
                text = t("Found {images} images. {captions} of them have a .txt caption, "
                         "but captions are taken from the file name.").format(
                    images=info.image_count, captions=info.caption_count)
                color = "#dc3545"
            else:
                text, color = t("Found {images} images.").format(images=info.image_count), "#198754"
        self.dataset_info_label.setText(text)
        self.dataset_info_label.setStyleSheet(f"color: {color};")

    def _update_steps_info(self, *_):
        if self._image_count == 0:
            self.steps_label.setText(t("Total steps are shown once the image folder is set."))
            return
        steps = self.controller.estimate_steps(
            self._image_count, self.repeats_spin.value(), self.batch_spin.value(), self.epochs_spin.value())
        self.steps_label.setText(t("About {steps} training steps in total. LoRAs usually need 1000-3000.").format(steps=steps))

    def _update_output_info(self, *_):
        preset = self._current_preset()
        if preset is None:
            self.output_info_label.setText("")
            return
        name = self.output_name_edit.text().strip() or "?"
        path = os.path.join(self.output_dir_edit.text().strip(), name + preset.output_extension)
        self.output_info_label.setText(t("The result will be saved as: {path}").format(path=path))

    def _browse_base_model(self):
        path = QFileDialog.getExistingDirectory(self, t("Base model"), self.base_model_edit.text())
        if path:
            self.base_model_edit.setText(path)

    def _browse_transformer_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, t("Custom model file"), os.path.dirname(self.transformer_model_edit.text()),
            "Model (*.safetensors *.gguf);;All Files (*.*)")
        if path:
            self.transformer_model_edit.setText(path)

    def _browse_dataset(self):
        path = QFileDialog.getExistingDirectory(self, t("Image folder"), self.dataset_edit.text())
        if path:
            self.dataset_edit.setText(path)
            if not self.output_name_edit.text().strip() or self.output_name_edit.text() == SimpleTrainSettings.output_name:
                self.output_name_edit.setText(os.path.basename(os.path.normpath(path)))
            self._update_dataset_info()

    def _browse_output_dir(self):
        path = QFileDialog.getExistingDirectory(self, t("Output folder"), self.output_dir_edit.text())
        if path:
            self.output_dir_edit.setText(path)

    def _open_output_dir(self):
        path = os.path.abspath(self.output_dir_edit.text().strip() or ".")
        os.makedirs(path, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def _open_advanced(self):
        if self._is_training():
            return
        settings = self._collect_settings()
        if settings is None:
            return
        answer = QMessageBox.question(
            self, t("Advanced mode"),
            t("This replaces the last settings of the advanced UI with the settings from this page. Continue?"),
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            settings.save()
            self.controller.export_to_advanced(settings)
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, t("Advanced mode"), str(e))
            return
        subprocess.Popen([sys.executable, os.path.join("scripts", "train_ui_qt.py")])
        self.close()

    def _is_training(self) -> bool:
        return self.train_controller.training_thread is not None

    def _on_training_button(self):
        if self._is_training():
            self.train_controller.start_training()  # stops the running training
            return

        settings = self._collect_settings()
        if settings is None:
            return
        errors = self.controller.validate(settings)
        if errors:
            self.show_validation_errors(errors)
            return

        self.settings = settings
        settings.save()
        try:
            self.controller.build_config(settings)
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, t("Cannot Start Training"), str(e))
            return
        self.train_controller.start_training()

    def closeEvent(self, event):
        if self._is_training():
            QMessageBox.warning(
                self,
                t("Training in progress"),
                t("A training is currently running. Stop the training before closing the window."),
            )
            event.ignore()
            return
        self._save_settings()
        event.accept()

    # --- callbacks used by TrainUIController ---

    def save_default(self):
        pass  # settings are saved before the config is built

    def show_validation_errors(self, errors: list[str]):
        bullet_list = "\n".join(f"• {t(e)}" for e in errors)
        QMessageBox.critical(self, t("Cannot Start Training"),
                             t("Please fix the following errors before training:") + f"\n\n{bullet_list}")

    def confirm(self, title: str, message: str) -> bool:
        return True  # the only question asked is whether to clear the latent cache, which is always safe

    def get_cloud_reattach(self) -> bool:
        return False

    def sync_cloud_secrets(self):
        pass

    def schedule_on_main_thread(self, fn: Callable):
        QTimer.singleShot(0, self, fn)

    def on_update_status(self, status: str):
        self.schedule_on_main_thread(lambda: self.status_label.setText(t(status)))

    def on_training_started(self):
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self._set_training_button_style("running")
        self._set_form_enabled(False)

    def on_training_stopping(self):
        self._set_training_button_style("stopping")

    def on_training_stopped(self, error_caught: bool):
        self.eta_label.setText("")
        self._set_training_button_style("idle")
        self._set_form_enabled(True)
        if not error_caught and self.progress_bar.maximum() > 1 \
                and self.progress_bar.value() >= self.progress_bar.maximum():
            self.status_label.setText(t("Done! Saved to {path}").format(path=self.train_config.output_model_destination))

    def on_update_progress(self, epoch_step: int, max_step: int, epoch: int, max_epoch: int, eta_str: str | None):
        self.schedule_on_main_thread(lambda: self._do_update_progress(epoch_step, max_step, epoch, max_epoch, eta_str))

    def on_download_progress(self, done: int, total: int | None):
        def update():
            if total:
                # QProgressBar is int32, so count in MB
                self.progress_bar.setRange(0, total // 2**20)
                self.progress_bar.setValue(done // 2**20)
            else:
                self.progress_bar.setRange(0, 0)  # busy indicator
        self.schedule_on_main_thread(update)

    def _do_update_progress(self, epoch_step: int, max_step: int, epoch: int, max_epoch: int, eta_str: str | None):
        total = max(max_step * max_epoch, 1)
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(min(epoch * max_step + epoch_step, total))
        text = t("Epoch {epoch}/{max_epoch}").format(epoch=min(epoch + 1, max_epoch), max_epoch=max_epoch)
        if eta_str is not None:
            text += "  ·  " + t("ETA:") + f" {t(eta_str)}"
        self.eta_label.setText(text)

    # --- helpers ---

    def _set_form_enabled(self, enabled: bool):
        self.centralWidget().findChild(QScrollArea).widget().setEnabled(enabled)

    def _set_training_button_style(self, mode: str):
        styles = {
            "idle":     ("Start Training", True,  "#198754"),
            "running":  ("Stop Training",  True,  "#dc3545"),
            "stopping": ("Stopping...",    False, "#dc3545"),
        }
        text, enabled, bg = styles[mode]
        self.training_button.setText(t(text))
        self.training_button.setEnabled(enabled)
        self.training_button.setStyleSheet(
            f"QPushButton {{ background-color: {bg}; color: white; padding: 6px; }}"
            f"QPushButton:disabled {{ background-color: {bg}; color: white; }}"
        )
