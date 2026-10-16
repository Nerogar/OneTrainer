import sys

# Force matplotlib into sys.modules before PySide6/shiboken installs its import
# hooks.  Its dateutil->six.moves chain builds a synthetic module whose repr()
# crashes on Python 3.12.0 (AttributeError: no '_path'; fixed in 3.12.1) when
# shiboken's getsource() scans it, aborting UI startup.  Caching it now keeps
# six.moves off the scanner's path.
import matplotlib.pyplot  # noqa: F401, ICN001

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
