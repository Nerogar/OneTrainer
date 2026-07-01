import sys

# Force pydantic internals into sys.modules before PySide6/shiboken installs its
# import hooks.  Without this, shiboken's inspect.getsource() fires on a
# partially-initialized pydantic module, causing a circular import error.
import pydantic._internal._validators  # noqa: F401
from util.import_util import script_imports

script_imports()

from modules.ui.PySide6TrainUIView import PySide6TrainView
from modules.util.ui.pyside6_util import create_application


def main():
    app = create_application()
    window = PySide6TrainView()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
