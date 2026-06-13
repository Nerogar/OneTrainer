from modules.ui.BaseSchedulerParamsWindowView import BaseKvParamsView, BaseSchedulerParamsWindowView
from modules.ui.PySide6ConfigListView import PySide6ConfigListView
from modules.ui.SchedulerParamsWindowController import KvParamsController, SchedulerParamsWindowController
from modules.util.ui import pyside6_components
from modules.util.ui.PySide6UIState import PySide6UIState

from PySide6.QtWidgets import QDialog, QGridLayout, QPushButton, QScrollArea, QWidget


class PySide6KvParamsView(PySide6ConfigListView, BaseKvParamsView):
    def __init__(self, master, controller: KvParamsController, ui_state):
        PySide6ConfigListView.__init__(
            self, master, controller, ui_state,
            attr_name="scheduler_params",
            from_external_file=False,
            add_button_text="add parameter",
            is_full_width=True,
        )
        BaseKvParamsView.__init__(self, pyside6_components)

    def refresh_ui(self):
        self._create_element_list()

    def create_widget(self, master, element, i, open_command, remove_command, clone_command, save_command):
        return PySide6KvWidget(master, element, i, open_command, remove_command, clone_command, save_command)


class PySide6KvWidget(QWidget):
    def __init__(self, master, element, i, open_command, remove_command, clone_command, save_command):
        super().__init__(master)
        self.element = element
        self.ui_state = PySide6UIState(element)
        self.i = i
        self.save_command = save_command

        lo = pyside6_components._layout(self)
        lo.setColumnStretch(1, 1)
        lo.setColumnStretch(2, 1)

        pyside6_components.colored_icon_button(self, 0, 0, "X", "#C00000", lambda: remove_command(self.i))

        # Key
        self.key = pyside6_components.entry(self, 0, 1, self.ui_state, "key",
                                            tooltip="Key name for an argument in your scheduler",
                                            wide_tooltip=True, width=50)
        self.key.editingFinished.connect(save_command)

        # Value
        self.value = pyside6_components.entry(self, 0, 2, self.ui_state, "value",
                                              tooltip="Value for an argument in your scheduler. Some special values can be used, wrapped in percent signs: LR, EPOCHS, STEPS_PER_EPOCH, TOTAL_STEPS, SCHEDULER_STEPS. Note that OneTrainer calls step() after every individual learning step, not every epoch, so what Torch calls 'epoch' you should treat as 'step'.",
                                              wide_tooltip=True, width=50)
        self.value.editingFinished.connect(save_command)

    def place_in_list(self):
        pyside6_components._layout(self.parent()).addWidget(self, getattr(self, 'visible_index', self.i), 0)
        self.show()

    def destroy(self):
        self.deleteLater()


class PySide6SchedulerParamsWindowView(BaseSchedulerParamsWindowView, QDialog):
    def __init__(self, parent, controller: SchedulerParamsWindowController, ui_state):
        QDialog.__init__(self, parent)
        BaseSchedulerParamsWindowView.__init__(self, pyside6_components)

        self.setWindowTitle("Learning Rate Scheduler Settings")
        self.resize(800, 500)

        outer = QGridLayout(self)
        outer.setRowStretch(0, 1)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        inner = QWidget()
        scroll.setWidget(inner)
        inner_lo = pyside6_components._layout(inner)
        inner_lo.setColumnStretch(1, 1)

        self.build_content(inner, controller, ui_state)

        expand_frame = QWidget(inner)
        inner_lo.addWidget(expand_frame, inner_lo.rowCount(), 0, 1, 2)
        # Must be assigned to an instance variable — PySide6ConfigListView is not a QWidget,
        # so Qt won't keep it alive. Without this, the GC collects it and the button's
        # clicked signal loses its connection to __add_element.
        self._kv_params_view = PySide6KvParamsView(expand_frame, KvParamsController(controller.config), ui_state)

        outer.addWidget(scroll, 0, 0)

        ok = QPushButton("ok", self)
        ok.clicked.connect(self.accept)
        outer.addWidget(ok, 1, 0)
