from modules.ui.BaseTimestepDistributionWindowView import BaseTimestepDistributionWindowView
from modules.ui.TimestepDistributionWindowController import TimestepDistributionWindowController
from modules.util.ui import pyside6_components

from matplotlib import pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from PySide6.QtWidgets import QDialog, QGridLayout, QPushButton


class PySide6TimestepDistributionWindowView(BaseTimestepDistributionWindowView, QDialog):
    def __init__(self, parent, controller: TimestepDistributionWindowController, ui_state):
        QDialog.__init__(self, parent)
        BaseTimestepDistributionWindowView.__init__(self, pyside6_components)

        # delete on close so entry widgets and the field validators they register globally are freed, not leaked
        self.finished.connect(self.deleteLater)

        self.setWindowTitle("Timestep Distribution")
        self.resize(900, 600)
        self._controller = controller

        outer = QGridLayout(self)
        outer.setRowStretch(0, 1)

        scroll, frame = pyside6_components.scrollable_frame(self)
        lo = pyside6_components._layout(frame)
        lo.setColumnStretch(3, 1)

        self.build_content(frame, controller, ui_state)
        lo.setRowStretch(7, 1)

        fig, self._ax = plt.subplots()
        self._canvas = FigureCanvasQTAgg(fig)

        palette = self.palette()
        text_color = palette.text().color().name()
        background_color = palette.window().color().name()
        fig.set_facecolor(background_color)

        self._ax.set_facecolor(self.palette().window().color().name())
        self._ax.spines['bottom'].set_color(text_color)
        self._ax.spines['left'].set_color(text_color)
        self._ax.spines['top'].set_color(text_color)
        self._ax.spines['right'].set_color(text_color)
        self._ax.tick_params(axis='x', colors=text_color, which="both")
        self._ax.tick_params(axis='y', colors=text_color, which="both")
        self._ax.xaxis.label.set_color(text_color)
        self._ax.yaxis.label.set_color(text_color)
        lo.addWidget(self._canvas, 0, 3, 8, 1)
        self._update_preview()

        update_btn = QPushButton("Update Preview", frame)
        update_btn.clicked.connect(self._update_preview)
        lo.addWidget(update_btn, 8, 3)

        outer.addWidget(scroll, 0, 0)

        ok = QPushButton("ok", self)
        ok.clicked.connect(self.accept)
        outer.addWidget(ok, 1, 0)


    def _update_preview(self):
        self._ax.cla()
        self._ax.hist(self._controller.generate_preview_data(), bins=1000, range=(0, 999))
        self._canvas.draw()
