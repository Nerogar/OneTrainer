from util.import_util import script_imports

script_imports()

import os
import sys

from modules.ui.controllers.windows.MaskController import MaskController
from modules.ui.utils.OneTrainerApplication import OnetrainerApplication
from modules.ui.utils.SNLineEdit import SNLineEdit

from PySide6.QtUiTools import QUiLoader


def main():
    os.environ["QT_QPA_PLATFORM"] = "xcb"  # Suppress Wayland warnings on NVidia drivers.
    # TODO: scalene (modules.ui.models.StateModel) changes locale on import, change QT6 locale to suppress warning here?

    app = OnetrainerApplication(sys.argv)
    loader = QUiLoader()
    loader.registerCustomWidget(SNLineEdit)

    onetrainer = MaskController(loader)

    # Invalidate ui elements after the controllers are set up, but before showing them.
    app.stateChanged.emit()
    onetrainer.ui.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
