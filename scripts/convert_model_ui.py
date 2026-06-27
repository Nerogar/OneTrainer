import sys

import pydantic._internal._validators  # noqa: F401
from util.import_util import script_imports

script_imports()

from modules.ui.ConvertModelUIController import ConvertModelUIController
from modules.ui.PySide6ConvertModelUIView import PySide6ConvertModelUIView

from PySide6.QtWidgets import QApplication


def main():
    _app = QApplication(sys.argv)
    ui = ConvertModelUIController().create_window(None, PySide6ConvertModelUIView)
    ui.exec()


if __name__ == '__main__':
    main()
