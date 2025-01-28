"""
`caption_ui.py` is designed to provide a graphical user interface (GUI) for image captioning tasks.

It uses the `CaptionUI` class to create the user interface.
This script is a standalone application, it can function independently of other scripts, although it may be integrated into larger workflows.
It can work in conjunction with scripts like `generate_captions.py` to facilitate image captioning.
It is used to allow users to quickly caption images.
"""
from util.import_util import script_imports

script_imports()

from modules.ui.CaptionUI import CaptionUI
from modules.util.args.CaptionUIArgs import CaptionUIArgs


def main():
    """
    Starts the Caption UI.

    Parses command line arguments using CaptionUIArgs.
    Initializes and starts the CaptionUI.
    """
    args = CaptionUIArgs.parse_args()

    ui = CaptionUI(None, args.dir, args.include_subdirectories)
    ui.mainloop()


if __name__ == '__main__':
    main()
