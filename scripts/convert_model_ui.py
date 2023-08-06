import os
import sys

sys.path.append(os.getcwd())

from modules.ui.ConvertModelUI import ConvertModelUI


def main():
    ui = ConvertModelUI(None)
    ui.mainloop()


if __name__ == '__main__':
    main()
