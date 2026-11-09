from modules.ui.TimestepDistributionWindowController import TimestepDistributionWindowController
from modules.util.ui import pyside6_components

from matplotlib import pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from PySide6.QtWidgets import QDialog, QGridLayout, QPushButton


class TimestepDistributionWindowView(QDialog):
    def __init__(self, parent, controller: TimestepDistributionWindowController, ui_state):
        QDialog.__init__(self, parent)
        self.components = pyside6_components

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

    def build_content(self, frame, controller, ui_state):
        # timestep distribution
        self.components.label(frame, 0, 0, "Timestep Distribution",
                         tooltip="Selects the function to sample timesteps during training",
                         wide_tooltip=True)
        self.components.options(frame, 0, 1, controller.get_distribution_options(), ui_state,
                           "timestep_distribution")

        # min noising strength
        self.components.label(frame, 1, 0, "Min Noising Strength",
                         tooltip="Specifies the minimum noising strength used during training. This can help to improve composition, but prevents finer details from being trained")
        self.components.entry(frame, 1, 1, ui_state, "min_noising_strength")

        # max noising strength
        self.components.label(frame, 2, 0, "Max Noising Strength",
                         tooltip="Specifies the maximum noising strength used during training. This can be useful to reduce overfitting, but also reduces the impact of training samples on the overall image composition")
        self.components.entry(frame, 2, 1, ui_state, "max_noising_strength")

        # noising weight
        self.components.label(frame, 3, 0, "Noising Weight",
                         tooltip="Controls the weight parameter of the timestep distribution function. Use the preview to see more details.")
        self.components.entry(frame, 3, 1, ui_state, "noising_weight")

        # noising bias
        self.components.label(frame, 4, 0, "Noising Bias",
                         tooltip="Controls the bias parameter of the timestep distribution function. Use the preview to see more details.")
        self.components.entry(frame, 4, 1, ui_state, "noising_bias")

        # timestep shift
        self.components.label(frame, 5, 0, "Timestep Shift",
                         tooltip="Shift the timestep distribution. Use the preview to see more details.")
        self.components.entry(frame, 5, 1, ui_state, "timestep_shift")

        # dynamic timestep shifting
        self.components.label(frame, 6, 0, "Dynamic Timestep Shifting",
                         tooltip="Dynamically shift the timestep distribution based on resolution. If enabled, the shifting parameters are taken from the model's scheduler configuration and Timestep Shift is ignored. Dynamic Timestep Shifting is not shown in the preview. For Ideogram, the shifting instead follows the model's own resolution-aware sampling schedule. Note: For Z-Image, the dynamic shifting parameters are likely wrong and unknown. Use with care or set your own, fixed shift.", wide_tooltip=True)
        self.components.switch(frame, 6, 1, ui_state, "dynamic_timestep_shifting")
