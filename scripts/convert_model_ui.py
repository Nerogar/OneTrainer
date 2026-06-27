from util.import_util import script_imports

script_imports()

from modules.ui.ConvertModelUIController import ConvertModelUIController
from modules.ui.CtkConvertModelUIView import CtkConvertModelUIView

import customtkinter as ctk


def main():
    # CtkConvertModelUIView is a CTkToplevel, so it needs a CTk root. Hide the root and run the
    # toplevel as a standalone window, tearing the root down once the window is closed.
    root = ctk.CTk()
    root.withdraw()
    ui = ConvertModelUIController().create_window(root, CtkConvertModelUIView)
    root.wait_window(ui)
    root.destroy()


if __name__ == '__main__':
    main()
