from util.import_util import script_imports

script_imports()

from modules.ui.CaptionUIController import CaptionUIController
from modules.ui.CtkCaptionUIView import CtkCaptionUIView
from modules.util.args.CaptionUIArgs import CaptionUIArgs

import customtkinter as ctk


def main():
    args = CaptionUIArgs.parse_args()

    # CtkCaptionUIView is a CTkToplevel, so it needs a CTk root. Hide the root and run the
    # toplevel as a standalone window, tearing the root down once the window is closed.
    root = ctk.CTk()
    root.withdraw()
    ui = CaptionUIController(args.dir, args.include_subdirectories).create_window(root, CtkCaptionUIView)
    root.wait_window(ui)
    root.destroy()


if __name__ == '__main__':
    main()
