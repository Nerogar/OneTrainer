from modules.ui.BaseConceptTabView import BaseConceptTabView, BaseConceptWidgetView
from modules.ui.ConceptTabController import ConceptTabController
from modules.ui.PySide6ConceptWindowView import PySide6ConceptWindowView
from modules.ui.PySide6ConfigListView import PySide6ConfigListView
from modules.util.ui import pyside6_components
from modules.util.ui.PySide6UIState import PySide6UIState
from modules.util.ui.QtVar import QtVar

from PIL.ImageQt import ImageQt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QCheckBox, QComboBox, QHBoxLayout, QLabel, QLineEdit, QPushButton, QWidget


class PySide6ConceptTabView(PySide6ConfigListView, BaseConceptTabView):

    def __init__(self, master, controller: ConceptTabController, ui_state):
        # Pre-initialize before PySide6ConfigListView.__init__ because _reset_filters is
        # called during build() via options_kv's initial command fire.
        self.search_var = QtVar("")
        self.filter_var = QtVar("ALL")
        self.show_disabled_var = QtVar(True)

        PySide6ConfigListView.__init__(
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
        return self.controller.open_element_window(self.master, self.current_config[i], ui_state[0], ui_state[1], ui_state[2], PySide6ConceptWindowView)

    def create_widget(self, master, element, i, open_command, remove_command, clone_command, save_command):
        return PySide6ConceptWidgetView(master, element, i, open_command, remove_command, clone_command, save_command, self.controller)

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
        filter_combo = QComboBox(toolbar)
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


class PySide6ConceptWidgetView(BaseConceptWidgetView, QWidget):

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
        self.name_label.setWordWrap(True)
        self.name_label.setFixedWidth(140)
        self.name_label.move(5, 153)

        close_btn = QPushButton("X", self)
        close_btn.setFixedSize(24, 24)
        close_btn.setStyleSheet("background-color: #C00000; color: white;")
        close_btn.move(5, 0)
        close_btn.clicked.connect(lambda: remove_command(self.i))

        clone_btn = QPushButton("+", self)
        clone_btn.setFixedSize(24, 24)
        clone_btn.setStyleSheet("background-color: #00C000; color: white;")
        clone_btn.move(34, 0)
        clone_btn.clicked.connect(lambda: clone_command(self.i, controller.randomize_seed))

        enabled_cb = QCheckBox(self)
        enabled_cb.setChecked(concept.enabled)
        enabled_cb.setFixedSize(20, 20)
        enabled_cb.setStyleSheet("QCheckBox::indicator { width: 20px; height: 20px; }")
        enabled_cb.move(135, 0)
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
