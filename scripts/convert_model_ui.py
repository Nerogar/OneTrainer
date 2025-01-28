"""
The `convert_model_ui.py` script is responsible for launching a graphical user interface (GUI) for model conversion.

It utilizes the `ConvertModelUI` class to instantiate the user interface.
This script is a standalone component that empowers users to interactively manage model conversions.
It's tightly coupled with `convert_model.py`, which handles the backend logic of model format transformations.
It provides a user-friendly way to change model formats.
"""
from util.import_util import script_imports

script_imports()

from modules.ui.ConvertModelUI import ConvertModelUI


def main():
    """
    Starts the Convert Model UI.

    Initializes and starts the ConvertModelUI.
    """
    ui = ConvertModelUI(None)
    ui.mainloop()


if __name__ == '__main__':
    main()
