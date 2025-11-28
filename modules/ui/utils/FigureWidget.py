from modules.ui.utils.MaskDrawingToolbar import MaskDrawingToolbar
from modules.util.enum.EditMode import EditMode
from modules.util.enum.MouseButton import MouseButton
from modules.util.enum.ToolType import ToolType

import PySide6.QtGui as QtG
from matplotlib.backend_bases import MouseButton as MplMouseButton
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import QCoreApplication as QCA
from PySide6.QtCore import Signal

# This class creates a FigureWidget associated with a matplotlib drawing area (self.figure) and a toolbar.
# The toolbar can contain a set of default tools for zooming (zoom_tools=True), and arbitrary tools specified as a list of dictionaries with the following keys:
# "type" (mandatory): tool type defined in modules.util.enum.ToolType
# "fn" (optional): callback invoked when the tool is used (clicked signal for buttons, valueChanged for spinboxes)
# "tool" (only for CHECKABLE_BUTTON): modules.util.enum.EditMode value associated with the tool. It will handle mutual exclusion of tools automatically
# "text" (optional): the tool's text
# "icon" (optional): the tool's icon
# "tooltip" (optional): the tool's tooltip text
# "shortcut" (optional): the tool's shortcut
# "name" (mandatory for spinboxes): the tool's QWidget objectName, to be used with findChild()
# "spinbox_range" (optional, for spinboxes only): (minimum, maximum, stepSize) for the spinbox
# "value" (optional, for spinboxes only): the spinbox' default value
#
# The FigureWidget has two event handling mechanisms:
# - Mutually exclusive tools (i.e., those associated with a ToolType.CHECKABLE_BUTTON defining a "tool" field) can use registerTool() to automatically invoke functions when clicked, released or mouseMoved events are fired.
# - The class can optionally emit QT6 signals (emit_clicked, emit_released, emit_wheel, emit_moved passed to __init__) which can be handled externally.
# RegisterTool accepts two types of callbacks:
# - use_mpl_event=True: callbacks will receive matplotlib events
# - use_mpl_event=False: callbacks will receive the same interface of QT6 signals.
#
# The QT6 signals exposed are:
# wheelUp()
# wheelDown()
# clicked(modules.utils.enum.MouseButton, int_x, int_y)
# released(modules.utils.enum.MouseButton, int_x, int_y)
# moved(modules.utils.enum.MouseButton, int_start_x, int_start_y, int_end_x, int_end_y)
# Coordinates are either in absolute canvas pixels (use_data_coordinates=False), or referred to the data loaded on the canvas (e.g., image or plot coordinates)

class FigureWidget(FigureCanvas):
    clicked = Signal(MouseButton, int, int) # x, y
    released = Signal(MouseButton, int, int) # x, y
    wheelUp = Signal()
    wheelDown = Signal()
    moved = Signal(MouseButton, int, int, int, int) # x0, y0, x1, y1.
    # Note: signals cannot be declared with unions like "int | None". So we either declare them as object to allow emitting None values, or use -1 for events outside the image (the latter approach is safer).

    def __init__(self, parent=None, width=5, height=4, dpi=100, zoom_tools=False, other_tools=None, emit_clicked=False, emit_released=False, emit_wheel=False, emit_moved=False, use_data_coordinates=True): # TODO: maybe add preferred size? Or use width/height in pixel and change figsize as the closest integer ratio (*10 // 10)?
        super().__init__(Figure(figsize=(width, height), layout="tight", dpi=dpi))
        self.toolbar = MaskDrawingToolbar(self, parent=parent)
        self.event_handlers = {}
        self.theme = "dark" if QtG.QGuiApplication.styleHints().colorScheme() == QtG.Qt.ColorScheme.Dark else "light"

        tools = []
        if zoom_tools:
            tools.extend([{
                "type": ToolType.BUTTON,
                "fn": self.toolbar.home,
                "icon": f"resources/icons/buttons/{self.theme}/house.svg",
                "tooltip": QCA.translate("toolbar_item", "Reset original view (CTRL+H)"),
                "shortcut": "Ctrl+H",
            },
            {
                "type": ToolType.CHECKABLE_BUTTON,
                "tool": EditMode.PAN,
                "icon": f"resources/icons/buttons/{self.theme}/move.svg",
                "tooltip": QCA.translate("toolbar_item", "Left button pans, Right button zooms (CTRL+P)"),
                "shortcut": "Ctrl+P",
            },
            {
                "type": ToolType.CHECKABLE_BUTTON,
                "tool": EditMode.ZOOM,
                "icon": f"resources/icons/buttons/{self.theme}/search.svg",
                "tooltip": QCA.translate("toolbar_item", "Zoom to rectangle (CTRL+Q)"),
                "shortcut": "Ctrl+Q",
            }])

        if other_tools is not None:
            if zoom_tools:
                tools.append({"type": ToolType.SEPARATOR})
            tools.extend(other_tools)

        self.toolbar.addTools(tools)

        if zoom_tools:
            self.registerTool(EditMode.PAN, clicked_fn=self.toolbar.press_pan, released_fn=self.toolbar.release_pan, use_mpl_event=True)
            self.registerTool(EditMode.ZOOM, clicked_fn=self.toolbar.press_zoom, released_fn=self.toolbar.release_zoom, use_mpl_event=True)

        self.use_data_coordinates = use_data_coordinates
        self.last_x = self.last_y = None


        self.emit_clicked = emit_clicked
        self.emit_released = emit_released
        self.emit_wheel = emit_wheel
        self.emit_moved = emit_moved

        self.mpl_connect("button_press_event", self.__eventHandler())
        self.mpl_connect("button_release_event", self.__eventHandler())
        self.mpl_connect("scroll_event", self.__eventHandler())
        self.mpl_connect("motion_notify_event", self.__eventHandler())

    def __eventHandler(self):
        def f(event):
            if event.name == "button_press_event":
                args = self.__onClicked(event)

                for k, v in self.event_handlers.items():
                    if k == str(self.toolbar.mode) and v["clicked"] is not None:
                        if v["use_mpl_event"]:
                            v["clicked"](event)
                        else:
                            v["clicked"](*args)


            elif event.name == "button_release_event":
                args = self.__onReleased(event)

                for k, v in self.event_handlers.items():
                    if k == str(self.toolbar.mode) and v["released"] is not None:
                        if v["use_mpl_event"]:
                            v["released"](event)
                        else:
                            v["released"](*args)

            elif event.name == "scroll_event":
                self.__onWheel(event)
            elif event.name == "motion_notify_event":
                args = self.__onMoved(event)

                for k, v in self.event_handlers.items():
                    if k == str(self.toolbar.mode) and v["moved"] is not None:
                        if v["use_mpl_event"]:
                            v["moved"](event)
                        else:
                            v["moved"](*args)

        return f


    def registerTool(self, tool_mode, clicked_fn=None, released_fn=None, moved_fn=None, use_mpl_event=False):
        self.event_handlers[str(tool_mode)] = {"clicked": clicked_fn, "released": released_fn, "moved": moved_fn, "use_mpl_event": use_mpl_event}


    # Process matplotlib event into a more abstract interface, and optionally emit a signal.
    def __onClicked(self, event):
        if self.use_data_coordinates:
            x, y = event.xdata, event.ydata
        else:
            x, y = event.x, event.y
        if event.button == MplMouseButton.LEFT:
            btn = MouseButton.LEFT
        elif event.button == MplMouseButton.RIGHT:
            btn = MouseButton.RIGHT
        elif event.button == MplMouseButton.MIDDLE:
            btn = MouseButton.MIDDLE
        else:
            btn = MouseButton.NONE

        self.last_x, self.last_y = x, y

        x = int(x) if x is not None else -1
        y = int(y) if y is not None else -1

        if self.emit_clicked:
            self.clicked.emit(btn, x, y)

        return btn, x, y


    def __onReleased(self, event):
        if self.use_data_coordinates:
            x, y = event.xdata, event.ydata
        else:
            x, y = event.x, event.y
        if event.button == MplMouseButton.LEFT:
            btn = MouseButton.LEFT
        elif event.button == MplMouseButton.RIGHT:
            btn = MouseButton.RIGHT
        elif event.button == MplMouseButton.MIDDLE:
            btn = MouseButton.MIDDLE
        else:
            btn = MouseButton.NONE

        self.last_x = self.last_y = None

        x = int(x) if x is not None else -1
        y = int(y) if y is not None else -1

        if self.emit_released:
            self.released.emit(btn, x, y)

        return btn, x, y


    def __onMoved(self, event):
        if self.use_data_coordinates:
            x1, y1 = event.xdata, event.ydata
        else:
            x1, y1 = event.x, event.y
        if event.button == MplMouseButton.LEFT:
            btn = MouseButton.LEFT
        elif event.button == MplMouseButton.RIGHT:
            btn = MouseButton.RIGHT
        elif event.button == MplMouseButton.MIDDLE:
            btn = MouseButton.MIDDLE
        else:
            btn = MouseButton.NONE

        x0, y0 = self.last_x, self.last_y

        self.last_x, self.last_y = x1, y1

        x0 = int(x0) if x0 is not None else -1
        y0 = int(y0) if y0 is not None else -1
        x1 = int(x1) if x1 is not None else -1
        y1 = int(y1) if y1 is not None else -1

        if self.emit_moved:
            self.moved.emit(btn, x0, y0, x1, y1) # If -1, either start or finish is outside the canvas.

        return btn, x0, y0, x1, y1

    def __onWheel(self, event):
        if self.emit_wheel:
            if event.button == "up":
                self.wheelUp.emit()
            elif event.button == "down":
                self.wheelDown.emit()

        return event.button
