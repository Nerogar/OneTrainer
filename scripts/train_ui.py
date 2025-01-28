"""
`train_ui.py` is responsible for launching a graphical user interface (GUI) for model training.

It uses the `TrainUI` class to instantiate the training user interface.
This script is a standalone component, it allows for interactive training management.
"""
from util.import_util import script_imports

script_imports()

from modules.ui.TrainUI import TrainUI


def main():
    """
    Starts the Train UI.

    Initializes and starts the TrainUI.
    Closes the UI after the main loop finishes.
    """
    ui = TrainUI()
    ui.mainloop()
    ui.close()


if __name__ == '__main__':
    main()
