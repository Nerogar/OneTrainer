import pydantic._internal._validators  # noqa: F401
from util.import_util import script_imports

script_imports()

from modules.ui.ConvertModelUIController import ConvertModelUIController
from modules.ui.PySide6ConvertModelUIView import PySide6ConvertModelUIView
from modules.util.ui.pyside6_util import create_application


def main():
    _app = create_application()
    ui = ConvertModelUIController().create_window(None, PySide6ConvertModelUIView)
    ui.exec()


if __name__ == '__main__':
    main()
