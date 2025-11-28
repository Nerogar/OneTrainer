import functools
import os
import re
import webbrowser

from modules.ui.models.StateModel import StateModel
from modules.ui.utils.SNLineEdit import SNLineEdit

import PySide6.QtCore as QtC
import PySide6.QtWidgets as QtW
from PySide6.QtCore import Qt, Slot
from showinfm import show_in_file_manager


# Abstract controller with some utility methods. Each controller is a Finite-State Machine executing:
# super()__init__ -> _setup -> _loadPresets -> _connectStateUI -> _connectUIBehavior -> _connectInputValidation -> _invalidateUI -> self.__init__
# After it is initialized, a controller reacts to external signals by using the slots connected with self._connect(), possibly with some helper methods.
# For legibility, the methods are grouped into: ###FSM###, ###Reactions###, ###Utils###
class BaseController:
    state_ui_connections = {} # Class attribute, but it will be overwritten by every subclass.

    def __init__(self, loader, ui_file, name=None, parent=None, **kwargs):
        self.loader = loader
        self.parent = parent
        self.ui = loader.load(ui_file, parentWidget=parent.ui if parent is not None else None)
        self.name = name

        self.connections = {}
        self.invalidation_callbacks = []

        self._setup()
        self._loadPresets()
        self._connectStateUI(self.state_ui_connections, StateModel.instance(), signal=QtW.QApplication.instance().stateChanged, update_after_connect=True, **kwargs)
        self._connectUIBehavior()
        self._connectInputValidation()
        self._invalidateUI()

    ###FSM###

    # Override this method to initialize auxiliary attributes of each controller.
    def _setup(self):
        pass

    # Override this method to load preset values for each control.
    def _loadPresets(self):
        pass

    # DO NOT override this method. It connects UI elements to SingletonConfigModel internal variables.
    # It will be automatically called for StateModel, but you should call it manually for other models.
    def _connectStateUI(self, connection_dict, model, signal=None, update_after_connect=False, group="global", **kwargs):
        for var, ui_names in connection_dict.items():
            if len(kwargs) > 0:
                var = var.format(**kwargs)

            if isinstance(ui_names, str):
                ui_names = [ui_names]
            for ui_name in ui_names:
                ui_elem = self.ui.findChild(QtC.QObject, ui_name)
                if ui_elem is None:
                    self._log("error", f"ERROR: {ui_name} not found.")
                else:
                    if isinstance(ui_elem, QtW.QCheckBox):
                        self._connect(ui_elem.stateChanged, self.__readCbx(ui_elem, var, model), group)
                    elif isinstance(ui_elem, QtW.QComboBox):
                        self._connect(ui_elem.activated, self.__readCbm(ui_elem, var, model), group)
                    elif isinstance(ui_elem, (QtW.QSpinBox, QtW.QDoubleSpinBox)):
                        self._connect(ui_elem.valueChanged, self.__readSbx(ui_elem, var, model), group)
                    elif isinstance(ui_elem, SNLineEdit): # IMPORTANT: keep this above base class!
                        self._connect(ui_elem.editingFinished, self.__readSNLed(ui_elem, var, model), group)
                    elif isinstance(ui_elem, QtW.QLineEdit):
                        self._connect(ui_elem.editingFinished, self.__readLed(ui_elem, var, model), group)

                    callback = functools.partial(BaseController._writeControl, ui_elem, var, model)
                    if signal is not None:
                        self._connect(signal, callback)
                    if update_after_connect:
                        callback()

    # Override this method to connect signals and slots intended for visual behavior (e.g., enable/disable controls).
    def _connectUIBehavior(self):
        pass

    # Override this method to handle complex field validation OTHER than the automatic validations defined in *.ui files.
    def _connectInputValidation(self):
        pass

    # DO NOT override this method. It triggers the UI updates queued by _connect(), at the end of super().__init__.
    def _invalidateUI(self):
        for fn, *args in self.invalidation_callbacks:
            if len(args) > 0 and args[0] is not None:
                fn(*args)
            else:
                fn()

    ###Reactions###

    # Connects a signal to a slot, possibly segregating it into a named category (for selectively disconnecting it later).
    # If update_after_connect is true, notifies the controller that the slot must be fired at the end of __init__. initial_args is a list of values to be passed during this initial firing.
    def _connect(self, signal_list, slot, key="global", update_after_connect=False, initial_args=None):
        if not isinstance(signal_list, list):
            signal_list = [signal_list]

        for signal in signal_list:
            if "DEBUG_UI" in os.environ:
                c = signal.connect(self.__debugSlot(key, signal, slot))
            else:
                c = signal.connect(slot)
            if key not in self.connections:
                self.connections[key] = []
            self.connections[key].append(c)

        # Schedule every update to be executed at the end of __init__
        if update_after_connect:
            if "DEBUG_UI" in os.environ:
                if initial_args is None:
                    self.invalidation_callbacks.append((self.__debugSlot(key, "Initial invalidation", slot), None))
                else:
                    self.invalidation_callbacks.append((self.__debugSlot(key, "Initial invalidation", slot), *initial_args))
            else:
                if initial_args is None:
                    self.invalidation_callbacks.append((slot, None))
                else:
                    self.invalidation_callbacks.append((slot, *initial_args))

    # Disconnects all the UI connections.
    def _disconnectAll(self):
        for v in self.connections.values():
            for c in v:
                self.ui.disconnect(c)

        self.connections = {}

    # Selectively disconnects only the connections belonging to a specific key (e.g., concept indexes).
    def _disconnectGroup(self, key):
        if key in self.connections:
            for c in self.connections[key]:
                self.ui.disconnect(c)
            self.connections[key] = []

    def _updateProgress(self, elem):
        @Slot(dict)
        def f(data):
            if "value" in data and "max_value" in data:
                val = int(elem.minimum() + data["value"] / data["max_value"] * (
                            elem.maximum() - elem.minimum())) if data["max_value"] > elem.minimum() else elem.minimum()
                if isinstance(elem, QtW.QProgressBar):
                    elem.setValue(val)
                elif isinstance(elem, QtW.QLabel):
                    elem.setText(str(val))
        return f

    ###Utils###

    # Opens a file dialog window when tool_button is pressed, then populates edit_box with the returned value.
    # Filters for file extensions follow QT6 syntax.
    def _connectFileDialog(self, tool_button, edit_box, is_dir=False, save=False, title=None, filters=None):
        def f(elem):
            diag = QtW.QFileDialog()

            if is_dir:
                dir = None
                if os.path.isdir(elem.text()):
                    dir = elem.text()
                txt = diag.getExistingDirectory(parent=None, caption=title, dir=dir)
                if txt != "":
                    elem.setText(self._removeWorkingDir(txt))
                    elem.editingFinished.emit()
            else:
                file = None
                if os.path.exists(elem.text()):
                    file = self._removeWorkingDir(elem.text())

                if save:
                    txt, flt = diag.getSaveFileName(parent=None, caption=title, dir=file, filter=filters)
                    if txt != "":
                        elem.setText(self._removeWorkingDir(self._appendExtension(txt, flt)))
                        elem.editingFinished.emit()
                else:
                    txt, _ = diag.getOpenFileName(parent=None, caption=title, dir=file, filter=filters)
                    if txt != "":
                        elem.setText(self._removeWorkingDir(txt))
                        elem.editingFinished.emit()

        self._connect(tool_button.clicked, functools.partial(f, edit_box))

    # Log a message with the given severity.
    def _log(self, severity, message):
        # TODO: if you prefer a GUI text area, print on it instead: https://stackoverflow.com/questions/24469662/how-to-redirect-logger-output-into-pyqt-text-widget
        # In that case it is important to register a global logger widget (e.g. on a window with different tabs for each severity level)
        # For high severity, maybe an alertbox can also be opened automatically
        StateModel.instance().log(severity, message)


    # Slot wrapper logging the events fired.
    def __debugSlot(self, key, signal, slot):
        self._log("debug", f"Connected [{key}]: {signal} -> {slot}")
        def f(*args):
            self._log("debug", f"Fired [{key}]: {signal} -> {slot}")
            return slot(*args)
        return f

    # Open a subwindow.
    def _openWindow(self, controller, fixed_size=False):
        if fixed_size:
            controller.ui.setWindowFlag(Qt.WindowCloseButtonHint)
            controller.ui.setWindowFlag(Qt.WindowMaximizeButtonHint, on=False)
            controller.ui.setFixedSize(controller.ui.size())
        controller.ui.show()

    # Open an alert window. Remember to translate the messages.
    def _openAlert(self, title, message, type="about", buttons=QtW.QMessageBox.StandardButton.Ok):
        wnd = None
        if type == "about":
            QtW.QMessageBox.about(self.ui, title, message) # About has no buttons nor return values.
        elif type == "critical":
            wnd = QtW.QMessageBox.critical(self.ui, title, message, buttons=buttons)
        elif type == "information":
            wnd = QtW.QMessageBox.information(self.ui, title, message, buttons=buttons)
        elif type == "question":
            wnd = QtW.QMessageBox.question(self.ui, title, message, buttons=buttons)
        elif type == "warning":
            wnd = QtW.QMessageBox.warning(self.ui, title, message, buttons=buttons)

        return wnd

    # Open an URL in the default web browser.
    def _openUrl(self, url):
        webbrowser.open(url, new=0, autoraise=False)

    # Open a directory in the OS' file browser.
    def _browse(self, dir):
        if os.path.isdir(dir):
            show_in_file_manager(dir)


    def _appendExtension(self, file, filter):
        patterns = filter.split("(")[1].split(")")[0].split(", ")
        for p in patterns:
            if re.match(p.replace(".", "\\.").replace("*", ".*"), file): # If the file already has a valid extension, return it as is.
                return file

        if "*" not in patterns[0]: # The pattern is a fixed filename, returning it regardless of the user selected name.
            return patterns[0] # TODO: maybe returning folder/patterns[0] is more reasonable? In original code there is: path_modifier=lambda x: Path(x).parent.absolute() if x.endswith(".json") else x (removes file and returns base folder instead)
        else:
            return "{}.{}".format(file, patterns[0].split("*.")[1]) # Append the first valid extension to file.

    # These methods cannot use directly lambdas, because variable names would be reassigned within the loop.
    @staticmethod
    def __readCbx( ui_elem, var, model):
        return lambda: model.set_state(var, ui_elem.isChecked())

    @staticmethod
    def __readCbm(ui_elem, var, model):
        return lambda: model.set_state(var, ui_elem.currentData())

    @staticmethod
    def __readSbx(ui_elem, var, model):
        return lambda x: model.set_state(var, x)

    @staticmethod
    def __readSNLed(ui_elem, var, model):
        return lambda: model.set_state(var, float(ui_elem.text()))

    @staticmethod
    def __readLed(ui_elem, var, model):
        return lambda: model.set_state(var, ui_elem.text())

    @staticmethod
    def _writeControl(ui_elem, var, model, *args): # Discard possible signal arguments.
        ui_elem.blockSignals(True)
        val = model.get_state(var)
        if val is not None:
            if isinstance(ui_elem, QtW.QCheckBox):
                ui_elem.setChecked(val)
            elif isinstance(ui_elem, QtW.QComboBox):
                idx = ui_elem.findData(val)
                if idx != -1:
                    ui_elem.setCurrentIndex(idx)
            elif isinstance(ui_elem, (QtW.QSpinBox, QtW.QDoubleSpinBox)):
                ui_elem.setValue(float(val))
            elif isinstance(ui_elem, (SNLineEdit, QtW.QLineEdit)): # IMPORTANT: keep this above base class!
                ui_elem.setText(str(val))
        ui_elem.blockSignals(False)

    def _removeWorkingDir(self, txt):
        cwd = os.getcwd()
        if txt.startswith(cwd):
            out = txt[len(cwd) + 1:]
            if out == "":
                out = "."
            return out # Remove working directory and trailing slash.
        else:
            return txt

    def _appendWidget(self, list_widget, controller, self_delete_fn=None, self_clone_fn=None):
        item = QtW.QListWidgetItem(list_widget)
        item.setSizeHint(controller.ui.size())
        list_widget.addItem(item)
        list_widget.setItemWidget(item, controller.ui)

        if self_delete_fn is not None:
            self._connect(controller.ui.deleteBtn.clicked, self_delete_fn)

        if self_clone_fn is not None:
            self._connect(controller.ui.cloneBtn.clicked, self_clone_fn)
