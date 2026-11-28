import sys

# Load these before PySide6 installs its import hooks, as in train_ui_qt.
import matplotlib.pyplot  # noqa: F401, ICN001
import pydantic._internal._validators  # noqa: F401
from util.import_util import script_imports

script_imports()

from modules.ui.CaptionUIController import CaptionUIController
from modules.ui.CaptionUIView import CaptionUIView
from modules.util.args.CaptionUIArgs import CaptionUIArgs
from modules.util.ui.pyside6_util import create_application, set_windows_taskbar_icon


def main():
    args = CaptionUIArgs.parse_args()
    app = create_application()
    window = CaptionUIController(args.dir, args.include_subdirectories).create_window(None, CaptionUIView)
    set_windows_taskbar_icon(window)
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
