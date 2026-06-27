from util.import_util import script_imports

script_imports()

from modules.ui.CtkVideoToolUIView import CtkVideoToolUIView
from modules.ui.VideoToolUIController import VideoToolUIController

import customtkinter as ctk


def main():
    # CtkVideoToolUIView is a CTkToplevel, so it needs a CTk root. Hide the root and run the
    # toplevel as a standalone window, tearing the root down once the window is closed.
    root = ctk.CTk()
    root.withdraw()
    ui = VideoToolUIController().create_window(root, CtkVideoToolUIView)
    root.wait_window(ui)
    root.destroy()


if __name__ == '__main__':
    main()
