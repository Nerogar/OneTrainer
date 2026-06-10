from util.import_util import script_imports

script_imports()

from modules.ui.VideoToolUI import VideoToolUI


def main():
    ui = VideoToolUI(None)
    ui.mainloop()


if __name__ == '__main__':
    main()
