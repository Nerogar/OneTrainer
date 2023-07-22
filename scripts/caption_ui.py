import os
import sys

from modules.util.args.CaptionUIArgs import CaptionUIArgs

sys.path.append(os.getcwd())

from modules.ui.CaptionUI import CaptionUI


def main():
    args = CaptionUIArgs.parse_args()

    ui = CaptionUI(None, args.dir)
    ui.mainloop()


if __name__ == '__main__':
    main()
