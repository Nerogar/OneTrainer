import os
import sys

sys.path.append(os.getcwd())

from modules.ui.TrainUI import TrainUI


def main():
    ui = TrainUI()
    ui.mainloop()
    ui.close()


if __name__ == '__main__':
    main()
