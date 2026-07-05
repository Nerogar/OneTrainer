# Kept for backwards compatibility with external launchers (e.g. Stability
# Matrix) that invoke this file directly instead of start-ui.bat/.sh.
from train_ui_qt import main

if __name__ == '__main__':
    main()
