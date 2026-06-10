from util.import_util import script_imports

script_imports()

from modules.ui.ConvertModelUI import ConvertModelUI


def main():
    ui = ConvertModelUI(None)
    ui.mainloop()


if __name__ == '__main__':
    main()
