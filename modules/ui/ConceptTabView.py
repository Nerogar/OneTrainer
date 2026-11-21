import os
import pathlib
from pathlib import Path

from modules.ui.ConceptTabController import ConceptTabController
from modules.ui.ConceptWindowController import ConceptWindowController
from modules.ui.ConceptWindowView import ConceptWindowView
from modules.ui.ConfigListView import BaseConfigListView, ConfigListView
from modules.util import path_util
from modules.util.config.ConceptConfig import ConceptConfig
from modules.util.enum.ConceptType import ConceptType
from modules.util.image_util import load_image
from modules.util.ui import pyside6_components
from modules.util.ui.PySide6UIState import PySide6UIState
from modules.util.ui.QtVar import QtVar

from PIL import Image
from PIL.ImageQt import ImageQt
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QLineEdit, QPushButton, QWidget


class BaseConceptTabView(BaseConfigListView):

    _FILTER_TYPES = ["ALL", "STANDARD", "VALIDATION", "PRIOR_PREDICTION"]

    def _element_matches_filters(self, element):
        if not self.filters.get("show_disabled", True):
            if hasattr(element, 'enabled') and not element.enabled:
                return False

        search = self.filters.get("search", "").lower()
        if search:
            if not hasattr(element, '_search_cache'):
                cache = []
                try:
                    if getattr(element, 'name', None):
                        cache.append(element.name.lower())
                    p = getattr(element, 'path', None)
                    if p:
                        try:
                            cache.append(os.path.basename(p).lower())
                            cache.append(p.lower())
                        except (TypeError, AttributeError):
                            pass
                except (AttributeError, TypeError):
                    pass
                element._search_cache = cache
            if not any(search in text for text in getattr(element, '_search_cache', [])):
                return False

        type_filter = self.filters.get("type", "ALL")
        if type_filter != "ALL":
            if hasattr(element, 'type') and element.type:
                try:
                    return ConceptType(element.type).value == type_filter
                except (ValueError, AttributeError):
                    return False
            return False

        return True



class BaseConceptWidgetView:

    def __init__(self, components, concept: ConceptConfig):
        self.components = components
        self.concept = concept

    def _get_display_name(self):
        if self.concept.name:
            return self.concept.name
        elif self.concept.path:
            return os.path.basename(self.concept.path)
        else:
            return ""

    def _get_preview_image(self):
        preview_path = "resources/icons/icon.png"
        glob_pattern = "**/*.*" if getattr(self.concept, 'include_subdirectories', False) else "*.*"

        concept_path = ConceptWindowController.get_concept_path(getattr(self.concept, 'path', None))
        if concept_path:
            for path in pathlib.Path(concept_path).glob(glob_pattern):
                if any(part.startswith('.') for part in path.relative_to(concept_path).parent.parts):
                    continue
                extension = os.path.splitext(path)[1]
                if (path.is_file()
                        and path_util.is_supported_image_extension(extension)
                        and not path.name.endswith("-masklabel.png")
                        and not path.name.endswith("-condlabel.png")):
                    preview_path = path_util.canonical_join(concept_path, path)
                    break
        try:
            image = load_image(preview_path, convert_mode="RGBA")
        except OSError:
            image = Image.new("RGBA", (150, 150), (200, 200, 200, 255))
        size = min(image.width, image.height)
        image = image.crop((
            (image.width - size) // 2,
            (image.height - size) // 2,
            (image.width - size) // 2 + size,
            (image.height - size) // 2 + size,
        ))
        return image.resize((150, 150), Image.Resampling.BILINEAR)


class ConceptTabView(ConfigListView, BaseConceptTabView):

    def __init__(self, master, controller: ConceptTabController, ui_state):
        # Pre-initialize before ConfigListView.__init__ because _reset_filters is
        # called during build() via options_kv's initial command fire.
        self.search_var = QtVar("")
        self.filter_var = QtVar("ALL")
        self.show_disabled_var = QtVar(True)

        ConfigListView.__init__(
            self, master, controller, ui_state,
            from_external_file=True,
            attr_name="concept_file_name",
            config_dir="training_concepts",
            default_config_name="concepts.json",
            add_button_text="Add Concept",
            add_button_tooltip="Adds a new concept to the current config.",
            is_full_width=False,
            show_toggle_button=True,
        )
        self._add_search_bar()

    def open_element_window(self, i, ui_state):
        return self.controller.open_element_window(self.master, self.current_config[i], ui_state[0], ui_state[1], ui_state[2], ConceptWindowView)

    def create_widget(self, master, element, i, open_command, remove_command, clone_command, save_command):
        return ConceptWidgetView(master, element, i, open_command, remove_command, clone_command, save_command, self.controller)

    def _create_element_list_frame(self, master):
        frame = super()._create_element_list_frame(master)
        viewport = frame._scroll_area.viewport()
        viewport._drop_filter = pyside6_components._FileDropFilter(
            viewport, self._add_dropped_folders, Path.is_dir, multiple=True,
        )
        self.empty_label = QLabel(
            "No concept populated for this config, meaning your training will have no data. "
            "Click 'Add Concept' or drag a folder of images in to add a concept",
            frame,
        )
        self.empty_label.setWordWrap(True)
        self.empty_label.setAlignment(Qt.AlignCenter)
        pyside6_components._layout(frame).addWidget(self.empty_label, 0, 0, 1, 7)
        return frame

    def _add_dropped_folders(self, paths):
        for path in paths:
            new_element = self.controller.create_new_element()
            new_element.path = Path(path).as_posix()
            self.current_config.append(new_element)
        self.widgets_initialized = False
        self._create_element_list()
        self.save_current_config()

    def _update_widget_visibility(self):
        super()._update_widget_visibility()
        self.empty_label.setVisible(not self.current_config)
        layout = self.element_list.parentWidget().layout()
        layout.setStretch(0, int(not self.current_config))
        layout.setStretch(1, int(bool(self.current_config)))

    def _add_search_bar(self):
        toolbar = QWidget(self.top_frame)
        row_lo = QHBoxLayout(toolbar)
        row_lo.setContentsMargins(0, 0, 0, 0)
        pyside6_components._layout(self.top_frame).addWidget(toolbar, 0, 4)

        self.search_var = QtVar("")
        search_entry = QLineEdit(toolbar)
        search_entry.setPlaceholderText("Filter...")
        search_entry.setFixedWidth(200)
        row_lo.addWidget(QLabel("Search:", toolbar))
        row_lo.addWidget(search_entry)

        def _on_search(text):
            self.search_var.set(text)
            self._update_filters()
        search_entry.textChanged.connect(_on_search)
        self.search_var._bind_widget(lambda v: search_entry.setText(v))

        self.filter_var = QtVar("ALL")
        filter_combo = pyside6_components.NoScrollComboBox(toolbar)
        filter_combo.addItems(self._FILTER_TYPES)
        filter_combo.setFixedWidth(150)
        row_lo.addWidget(QLabel("Type:", toolbar))
        row_lo.addWidget(filter_combo)

        def _on_filter(text):
            self.filter_var.set(text)
            self._update_filters()
        filter_combo.currentTextChanged.connect(_on_filter)
        self.filter_var._bind_widget(lambda v: filter_combo.setCurrentText(v))

        self.show_disabled_var = QtVar(True)
        show_disabled_cb = QCheckBox("Show Disabled", toolbar)
        show_disabled_cb.setChecked(True)
        row_lo.addWidget(show_disabled_cb)

        def _on_show_disabled(state):
            self.show_disabled_var.set(bool(state))
            self._update_filters()
        show_disabled_cb.stateChanged.connect(_on_show_disabled)
        self.show_disabled_var._bind_widget(lambda v: show_disabled_cb.setChecked(bool(v)))

        clear_btn = QPushButton("Clear", toolbar)
        clear_btn.setFixedWidth(50)
        clear_btn.clicked.connect(self._reset_filters)
        row_lo.addWidget(clear_btn)

    def _update_filters(self):
        self._create_element_list(search=self.search_var.get(),
                                  type=self.filter_var.get(),
                                  show_disabled=self.show_disabled_var.get())

    def _reset_filters(self):
        if self.search_var is not None:
            self.search_var.set("")
        if self.filter_var is not None:
            self.filter_var.set("ALL")
        if self.show_disabled_var is not None:
            self.show_disabled_var.set(True)
        self._update_filters()


class ConceptWidgetView(BaseConceptWidgetView, QWidget):

    def __init__(self, master, concept, i, open_command, remove_command, clone_command, save_command, controller):
        QWidget.__init__(self, master)
        BaseConceptWidgetView.__init__(self, pyside6_components, concept)
        self.ui_state = PySide6UIState(concept)
        self.image_ui_state = PySide6UIState(concept.image)
        self.text_ui_state = PySide6UIState(concept.text)
        self.i = i

        self.setFixedSize(160, 180)

        image = self._get_preview_image()
        pixmap = QPixmap.fromImage(ImageQt(image.convert("RGBA")))
        self.image_label = QLabel(self)
        self.image_label.setPixmap(pixmap)
        self.image_label.setFixedSize(150, 150)
        self.image_label.move(5, 0)
        self.image_label.mousePressEvent = lambda _: open_command(
            self.i, (self.ui_state, self.image_ui_state, self.text_ui_state)
        )

        self.name_label = QLabel(self._get_display_name(), self)
        self.name_label.setFixedWidth(140)
        self.name_label.move(5, 153)

        control_margin = 4
        close_btn = pyside6_components.list_action_button(
            self, "remove", lambda: remove_command(self.i), size=24,
            tooltip="Deletes the concept",
        )
        close_btn.move(5 + control_margin, control_margin)

        clone_btn = pyside6_components.list_action_button(
            self, "clone", lambda: clone_command(self.i, controller.randomize_seed), size=24,
            tooltip="Duplicates the concept",
        )
        clone_btn.move(34 + control_margin, control_margin)

        enabled_cb = QCheckBox(self)
        enabled_cb.setChecked(concept.enabled)
        enabled_cb.setFixedSize(20, 20)
        enabled_cb.setProperty("largeIndicator", True)
        enabled_cb.move(135 - control_margin, control_margin)
        enabled_cb.stateChanged.connect(lambda state: (
            setattr(concept, 'enabled', bool(state)),
            save_command(),
        ))
        self.ui_state.get_var("enabled")._bind_widget(
            lambda v: enabled_cb.setChecked(bool(v))
        )

    def configure_element(self):
        self.name_label.setText(self._get_display_name())
        image = self._get_preview_image()
        pixmap = QPixmap.fromImage(ImageQt(image.convert("RGBA")))
        self.image_label.setPixmap(pixmap)
        try:
            if hasattr(self.concept, '_search_cache'):
                delattr(self.concept, '_search_cache')
        except AttributeError:
            pass

    def place_in_list(self):
        index = getattr(self, 'visible_index', self.i)
        x = index % 6
        y = index // 6
        lo = pyside6_components._layout(self.parent())
        lo.addWidget(self, y, x)
        lo.setColumnStretch(6, 1)
        self.show()

    def destroy(self):
        self.deleteLater()
