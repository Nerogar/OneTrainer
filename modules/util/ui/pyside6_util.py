import os
import signal
import sys
from abc import ABCMeta

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication, QStyleFactory, QWidget


class QtABCMeta(type(QWidget), ABCMeta):
    # Combined metaclass that resolves the conflict between Qt's Shiboken metaclass and ABCMeta.
    pass


def create_application() -> QApplication:
    # Restore the OS default SIGINT handler so Ctrl+C terminates the process
    # directly at the C level. Qt's event loop blocks inside C++, so Python's
    # own SIGINT handler would never get a chance to run while app.exec() is
    # active and Ctrl+C would be ignored.
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    # On desktops without the xdg-desktop-portal Settings interface, Qt spams two
    # "qt.qpa.theme.gnome: dbus reply error ... org.freedesktop.portal.Settings"
    # lines while probing for the system color scheme. Silence just that category;
    # the rules string is read when Qt's logging initializes at QApplication init.
    _gnome_theme_rule = "qt.qpa.theme.gnome=false"
    existing_rules = os.environ.get("QT_LOGGING_RULES")
    os.environ["QT_LOGGING_RULES"] = f"{existing_rules};{_gnome_theme_rule}" if existing_rules else _gnome_theme_rule

    app = QApplication(sys.argv)
    # Force Fusion everywhere: native styles (e.g. windowsvista) draw standard
    # controls via OS theme APIs, which breaks once an application stylesheet
    # is set, producing a flatter look than Fusion's own stylesheet-aware painting.
    app.setStyle(QStyleFactory.create("Fusion"))
    app.styleHints().setColorScheme(Qt.ColorScheme.Light)

    palette = app.palette()
    palette.setColor(QPalette.ColorRole.Base, QColor("white"))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Base, QColor("#e0e0e0"))
    app.setPalette(palette)

    app.setStyleSheet("""
        QLineEdit, QSpinBox, QDoubleSpinBox, QTextEdit, QPlainTextEdit {
            padding: 2px 2px;
        }
        QCheckBox::indicator {
            width: 16px;
            height: 16px;
        }
        QProgressBar {
            background-color: #c8c8c8;
        }
        QToolButton {
            padding-top: 0px;
            padding-bottom: 0px;
            padding-right: 40px;
        }
        QToolButton::menu-indicator {
            subcontrol-origin: padding;
            subcontrol-position: right center;
            width: 12px;
            height: 12px;
            right: 10px;
        }
    """)

    return app
