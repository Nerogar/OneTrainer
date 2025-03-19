import logging

from util.import_util import script_imports

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

script_imports()

from modules.ui.TrainUI import TrainUI


def main():
    ui = TrainUI()
    ui.mainloop()


if __name__ == '__main__':
    main()
