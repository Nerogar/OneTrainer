import os
import sys

sys.path.append(os.getcwd())

from modules.util.args.CaptionUIArgs import CaptionUIArgs
from modules.ui.CaptionUI import CaptionUI


def main():
    args = CaptionUIArgs.parse_args()

    ui = CaptionUI(None, args.dir)
    ui.mainloop()


if __name__ == '__main__':
    main()
