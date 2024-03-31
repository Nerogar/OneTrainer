from util.import_util import script_imports

script_imports()

from modules.ui.TrainUI import TrainUI


def main():
    ui = TrainUI()
    ui.mainloop()
    ui.close()


if __name__ == '__main__':
    main()
