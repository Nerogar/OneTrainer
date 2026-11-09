import os

from modules.util.enum.PathIOType import PathIOType
from modules.util.ui import pyside6_components as components
from modules.util.ui.pyside6_util import fit_form, limit_input_width
from modules.util.ui.PySide6UIState import PySide6UIState

from PySide6.QtCore import QEvent, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)


class PySide6QueueTrainingUIView(QDialog):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.state = PySide6UIState(controller.settings)
        self.finished.connect(self.deleteLater)
        self.setWindowTitle("Training Queue Generator")
        self.resize(950, 650)
        outer = QVBoxLayout(self)
        open_queue = QPushButton("Open saved queue")
        open_queue.setToolTip("Reopen existing queue for editing")
        open_queue.clicked.connect(self._open_queue)
        open_queue.ensurePolished()
        open_queue.setMaximumWidth(2 * open_queue.sizeHint().width())
        outer.addWidget(open_queue, alignment=Qt.AlignmentFlag.AlignLeft)
        scroll, content = components.scrollable_frame(self)
        outer.addWidget(scroll)
        layout = components._layout(content)
        general = QWidget(content)
        powers = ["none", "shutdown", "sleep"] + (["hibernate"] if os.name == "nt" else [])
        self.build_general(general, self.state, powers)
        layout.addWidget(general, 0, 0)
        self.loop_frame = QWidget(content)
        loop_layout = QVBoxLayout(self.loop_frame)
        loop_layout.addWidget(
            QLabel("Run exported configs in the listed order. Their validation settings are preserved.")
        )
        self.loop_table = self._table("Loop", ["Config path"], [""])
        loop_layout.addWidget(self.loop_table)
        controls = QHBoxLayout()
        for label, callback, tooltip in [
            ("Add configs", self._add_configs, "Add an exported training config"),
            ("Remove", lambda: self._remove(self.loop_table), "Remove selected queue entries; keep the config files."),
            ("Move up", lambda: self._move(-1), "Move the selected config up in the queue"),
            ("Move down", lambda: self._move(1), "Move the selected config down in the queue"),
        ]:
            button = QPushButton(label)
            button.setToolTip(tooltip)
            button.clicked.connect(callback)
            if label != "Add configs":
                button.setEnabled(False)
                self.loop_table.itemSelectionChanged.connect(
                    lambda button=button: button.setEnabled(bool(self.loop_table.selectedIndexes()))
                )
            controls.addWidget(button)
        loop_layout.addLayout(controls)
        layout.addWidget(self.loop_frame, 1, 0)
        self.source_frame = QWidget(content)
        self.build_source(self.source_frame, self.state)
        layout.addWidget(self.source_frame, 2, 0)
        self.parameter_frame = QWidget(content)
        parameter_layout = QVBoxLayout(self.parameter_frame)
        parameter_layout.addWidget(
            QLabel("Parameter keys use config names, e.g. learning_rate or optimizer.weight_decay.")
        )
        self.sweep_table = self._table("Sweep", ["Key", "Values (comma list, JSON list, or start:end:step)"], [
            "Config field to vary. Nested paths and list indices work, e.g. optimizer.weight_decay or additional_embeddings[0].model_name.",
            'Comma list, JSON list, or inclusive range (0.01:0.03:0.01). Quote strings in JSON, e.g. ["512", "1024"]. All combinations run.',
        ])
        parameter_layout.addWidget(self.sweep_table)
        controls = QHBoxLayout()
        add = QPushButton("Add parameter")
        add.setToolTip("Vary another config field. Every combination of parameter values creates a training run.")
        add.clicked.connect(self._add_parameter)
        remove = QPushButton("Remove parameter")
        remove.setToolTip("Stop varying the selected parameters; their base configuration values apply.")
        remove.clicked.connect(lambda: self._remove(self.sweep_table))
        remove.setEnabled(False)
        self.sweep_table.itemSelectionChanged.connect(lambda: remove.setEnabled(bool(self.sweep_table.selectedIndexes())))
        controls.addWidget(add)
        controls.addWidget(remove)
        parameter_layout.addLayout(controls)
        layout.addWidget(self.parameter_frame, 3, 0)
        export = self.export_button = QPushButton("Generate script")
        export.setAutoDefault(False)
        export.clicked.connect(self._export)
        font = export.font()
        font.setPointSizeF(font.pointSizeF() + 2)
        font.setBold(True)
        export.setFont(font)
        export.ensurePolished()
        export.setFixedWidth(2 * export.sizeHint().width())
        export.setMinimumHeight(round(1.4 * export.sizeHint().height()))
        actions = QHBoxLayout()
        actions.addStretch()
        self.overwrite = QCheckBox("Overwrite existing")
        self.overwrite.setToolTip("Replace existing files at the output script and companion JSON paths. Input configs are protected.")
        actions.addWidget(self.overwrite)
        actions.addWidget(export)
        outer.addLayout(actions)
        self.base_path_input.findChild(QLineEdit).textChanged.connect(self._update_export)
        trace = self.state.add_var_trace("mode", self._update_mode)
        self.destroyed.connect(lambda: self.state.remove_var_trace("mode", trace))
        source_trace = self.state.add_var_trace("source", self._update_source)
        self.destroyed.connect(lambda: self.state.remove_var_trace("source", source_trace))
        self._update_mode()
        self._update_source()

    def build_general(self, frame, state, power_options):
        tooltip = "Loop runs existing configs in order. Sweep copies a base config for every parameter combination, with separate outputs and caches."
        components.label(frame, 0, 0, "Mode", tooltip=tooltip)
        components.options(frame, 0, 1, ["Loop", "Sweep"], state, "mode").setToolTip(tooltip)
        for row, label, key, mode, tooltip in (
            (1, "Sweep output directory", "output_dir", "dir",
             "Each export creates a fresh sweep folder with separate configs, workspaces, caches, and model outputs. Automatic backup continuation is disabled."),
            (2, "Output script (.bat / .sh)", "script_path", "file",
             "The .bat runs training CLI sequentially, failure or cancellation stops the script. Rerunning reuses destinations. Generate again for fresh Sweep outputs."),
        ):
            label_widget = components.label(frame, row, 0, label, tooltip=tooltip)
            path_widget = components.path_entry(
                frame, row, 1, state, key, mode=mode, allow_model_files=False,
                io_type=PathIOType.OUTPUT,
            )
            path_widget.setToolTip(tooltip)
            entry = path_widget.findChild(QLineEdit)
            limit_input_width(entry, 30 if key == "script_path" else 40)
            path_widget.setFixedWidth(entry.maximumWidth() + path_widget.findChild(QPushButton).width())
            if key == "output_dir":
                self.output_widgets = (label_widget, path_widget)
        tooltip = "Actions apply *only* if training successfully completes"
        components.label(frame, 3, 0, "On successful completion", tooltip=tooltip)
        components.options(frame, 3, 1, power_options, state, "power_action").setToolTip(tooltip)
        fit_form(frame)

    def build_source(self, frame, state):
        tooltip = "Use current UI settings, a config file, or the snapshot from a reopened queue. Saved snapshots retain their concepts and samples."
        components.label(frame, 0, 0, "Base configuration", tooltip=tooltip)
        components.options(frame, 0, 1, ["Current UI", "Config file", "Saved snapshot"], state, "source").setToolTip(tooltip)
        tooltip = "Copy this config as the Sweep base. The source file stays unchanged."
        self.base_path_label = components.label(frame, 1, 0, "Config file", tooltip=tooltip)
        self.base_path_input = components.path_entry(frame, 1, 1, state, "base_path", allow_model_files=False)
        self.base_path_input.setToolTip(tooltip)
        entry = self.base_path_input.findChild(QLineEdit)
        limit_input_width(entry)
        self.base_path_input.setFixedWidth(entry.maximumWidth() + self.base_path_input.findChild(QPushButton).width())
        fit_form(frame)

    def _open_queue(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open saved queue", "", "JSON (*.json)")
        if not path:
            return
        try:
            self.controller.load(path)
        except Exception as exc:
            QMessageBox.warning(self, "Cannot open queue", str(exc))
            return
        self.state.update(self.controller.settings)
        for table in (self.loop_table, self.sweep_table):
            table.setRowCount(0)
            for data in table.queue_rows:
                self._append_row(table, data)
        self._update_mode()
        self._update_source()

    def _update_source(self):
        use_file = self.controller.settings["source"] == "Config file"
        self.base_path_label.setVisible(use_file)
        self.base_path_input.setVisible(use_file)
        self._update_export()

    def _update_export(self):
        # Check availability here; filesystem validation belongs to export, not UI updates.
        if self.controller.settings["mode"] == "Loop":
            paths = [self.loop_table.cellWidget(row, 0).findChild(QLineEdit).text().strip()
                     for row in range(self.loop_table.rowCount())]
            available = bool(paths) and all(paths)
        elif self.controller.settings["source"] == "Config file":
            available = bool(self.base_path_input.findChild(QLineEdit).text().strip())
        elif self.controller.settings["source"] == "Saved snapshot":
            available = self.controller.base_snapshot is not None
        else:
            available = self.controller.train_config is not None
        self.export_button.setEnabled(available)

    def _table(self, mode, headers, tooltips):
        table = QTableWidget(0, len(headers))
        table.queue_rows = self.controller.rows[mode]
        table.setHorizontalHeaderLabels(headers)
        for column, tooltip in enumerate(tooltips):
            table.horizontalHeaderItem(column).setToolTip(tooltip)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.verticalHeader().setStyleSheet("QHeaderView::section { padding-left: 4px; padding-right: 4px; }")
        table.setMinimumHeight(170)
        for data in table.queue_rows:
            self._append_row(table, data)
        return table

    def _append_row(self, table, data):
        row = table.rowCount()
        table.insertRow(row)
        state = PySide6UIState(data)
        for column, key in enumerate(data):
            cell = QWidget(table)
            entry = components.entry(cell, 0, 0, state, key, width=60, sticky="new",
                                     tooltip=table.horizontalHeaderItem(column).toolTip())
            entry.queue_table = table
            entry.installEventFilter(self)
            table.setCellWidget(row, column, cell)
            if key == "path":
                entry.textChanged.connect(self._update_export)
        table.resizeRowToContents(row)

    def eventFilter(self, watched, event):
        if event.type() in (QEvent.Type.FocusIn, QEvent.Type.MouseButtonPress):
            table = watched.queue_table
            index = table.indexAt(watched.parentWidget().pos())
            table.setCurrentCell(index.row(), index.column())
            table.selectRow(index.row())
        return super().eventFilter(watched, event)

    def _update_mode(self):
        mode = self.controller.settings["mode"]
        self.loop_frame.setVisible(mode == "Loop")
        self.source_frame.setVisible(mode != "Loop")
        self.parameter_frame.setVisible(mode != "Loop")
        self.sweep_table.setVisible(mode == "Sweep")
        for widget in self.output_widgets:
            widget.setVisible(mode == "Sweep")
        self._update_export()

    def _add_configs(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Choose exported configs", "", "JSON (*.json)",
            options=QFileDialog.Option.DontUseNativeDialog | QFileDialog.Option.DontUseCustomDirectoryIcons,
        )
        for path in paths:
            self._append_row(self.loop_table, self.controller.add_row("Loop", path=path))
        self._update_export()

    def _remove(self, table):
        table.setFocus()
        for row in sorted({index.row() for index in table.selectedIndexes()}, reverse=True):
            del table.queue_rows[row]
            table.removeRow(row)
        self._update_export()

    def _move(self, delta):
        row = self.loop_table.currentRow()
        other = row + delta
        if row >= 0 and 0 <= other < self.loop_table.rowCount():
            rows = self.controller.rows["Loop"]
            rows[row], rows[other] = rows[other], rows[row]
            self.loop_table.setRowCount(0)
            for data in rows:
                self._append_row(self.loop_table, data)
            self.loop_table.setCurrentCell(other, 0)

    def _add_parameter(self):
        self._append_row(self.sweep_table, self.controller.add_row("Sweep"))

    def _export(self):
        # Commit focused editor values before reading the bound state.
        if self.focusWidget():
            self.focusWidget().clearFocus()
        for entry in self.findChildren(QLineEdit):
            if entry.isVisibleTo(self) and entry.isEnabled() and hasattr(entry, "_validator"):
                error = entry._validator.flush()
                if error:
                    QMessageBox.warning(self, "Invalid field", error)
                    entry.setFocus()
                    return
        try:
            path = self.controller.export(overwrite=self.overwrite.isChecked())
        except Exception as exc:
            QMessageBox.warning(self, "Cannot generate queue", str(exc))
        else:
            QMessageBox.information(
                self, "Queue exported",
                f"Created script:\n{self.controller.settings['script_path']}\n\nSaved queue settings:\n{path}",
            )
