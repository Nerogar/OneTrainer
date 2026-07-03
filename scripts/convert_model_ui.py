import json
from contextlib import suppress

import pydantic._internal._validators  # noqa: F401
from util.import_util import script_imports

script_imports()

from modules.ui.ConvertModelUIController import ConvertModelUIController
from modules.ui.PySide6ConvertModelUIView import PySide6ConvertModelUIView
from modules.util.config.SecretsConfig import SecretsConfig
from modules.util.ui.pyside6_util import create_application


def main():
    _app = create_application()

    huggingface_token = ""
    with suppress(FileNotFoundError), open("secrets.json", "r") as f:
        secrets_dict = json.load(f)
        huggingface_token = SecretsConfig.default_values().from_dict(secrets_dict).huggingface_token

    ui = ConvertModelUIController(huggingface_token=huggingface_token).create_window(None, PySide6ConvertModelUIView)
    ui.exec()


if __name__ == '__main__':
    main()
