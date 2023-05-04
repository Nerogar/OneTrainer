import os
import sys

from modules.ui.TrainUI import TrainUI

sys.path.append(os.getcwd())


def main():
    ui = TrainUI()
    ui.mainloop()


if __name__ == '__main__':
    main()
