import pydantic._internal._validators  # noqa: F401
from util.import_util import script_imports

script_imports()

from modules.ui.VideoToolUIController import VideoToolUIController
from modules.ui.VideoToolUIView import VideoToolUIView
from modules.util.ui.pyside6_util import create_application


def main():
    _app = create_application()
    ui = VideoToolUIController().create_window(None, VideoToolUIView)
    ui.exec()


if __name__ == '__main__':
    main()
