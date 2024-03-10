import os
import sys

sys.path.append(os.getcwd())

from modules.ui.TrainUI import TrainUI
from modules.zluda.utils import fix_path


def main():
    fix_path()

    ui = TrainUI()
    ui.mainloop()
    ui.close()


if __name__ == '__main__':
    main()
