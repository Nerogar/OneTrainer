from util.import_util import script_imports

script_imports()

from modules.ui.CtkTrainUIView import CtkTrainUIView


def main():
    ui = CtkTrainUIView()
    ui.mainloop()


if __name__ == '__main__':
    main()
