from util.import_util import script_imports

script_imports()

from modules.ui.TrainUI import TrainUI
from modules.util.logging_util import silence_pyvips


def main():
    # Silence PyVips verbose logging at startup
    silence_pyvips(True)

    ui = TrainUI()
    ui.mainloop()


if __name__ == '__main__':
    main()
