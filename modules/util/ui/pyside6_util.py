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

    app = QApplication(sys.argv)
    # Force Fusion everywhere: native styles (e.g. windowsvista) draw standard
    # controls via OS theme APIs, which breaks once an application stylesheet
    # is set, producing a flatter look than Fusion's own stylesheet-aware painting.
    app.setStyle(QStyleFactory.create("Fusion"))

    color_schemes = {
        Qt.ColorScheme.Light: {
            "base": "#F9F9F9",
            "text": "#1C1C1C",
            "disabled": "#E0E0E0",
            "window": "#CFCFCF",
            "window_section": "#DBDBDB",
            "window_text": "#1C1C1C",
            "button": "#36719F",
            "button_hover": "#3B8ED0",
            "button_disabled": "#A0A0A0",
            "editbox_frame": "#999CA1",
            "progress_bar": "#939BA2",
            "checkbox": "#6AA5D2",
            "checkbox_hover": "#75B6E5",
            "checkbox_glyph": "checkbox_light.png",
        },
        Qt.ColorScheme.Dark: {
            "base": "#343638",
            "text": "#DCE4EE",
            "disabled": "#2D2D2D",
            "window": "#333333",
            "window_text": "#DCE4EE",
            "window_section": "#2B2B2B",
            "button": "#144870",
            "button_hover": "#195A8C",
            "button_disabled": "#404040",
            "editbox_frame": "#565B5E",
            "progress_bar": "#4A4D50",
            "checkbox": "#346185",
            "checkbox_hover": "#3D739C",
            "checkbox_glyph": "checkbox_dark.png",
        },
    }

    def apply_palette(scheme=Qt.ColorScheme.Dark):
        palette = app.palette()
        colors = color_schemes[scheme]
        palette.setColor(QPalette.ColorRole.Base, QColor(colors["base"]))
        palette.setColor(QPalette.ColorRole.Text, QColor(colors["text"]))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(colors["button_hover"]))
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Base, QColor(colors["disabled"]))
        palette.setColor(QPalette.ColorRole.Window, QColor(colors["window"]))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(colors["window_text"]))
        app.setPalette(palette)

        style_sheet = """
            QLineEdit, QSpinBox, QDoubleSpinBox, QTextEdit, QPlainTextEdit {{
                padding: 2px 2px;
                border: 2px solid {editbox_frame};
                border-radius: 4px;
            }}
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QTextEdit:focus, QPlainTextEdit:focus {{
                border: 2px solid palette(Highlight);
            }}
            QCheckBox::indicator {{
                width: 18px;
                height: 18px;
            }}
            QProgressBar {{
                background-color: {progress_bar};
            }}
            QToolButton {{
                padding-top: 0px;
                padding-bottom: 0px;
                padding-right: 40px;
            }}
            QToolButton::menu-indicator {{
                subcontrol-origin: padding;
                subcontrol-position: right center;
                width: 12px;
                height: 12px;
                right: 10px;
            }}
            QFrame#section_frame {{
                background-color: {window_section};
                border-radius: 4px;
            }}
        """
        if scheme == Qt.ColorScheme.Dark:
            style_sheet += """
                QToolButton, QPushButton, QComboBox {{
                    background: {button};
                }}
                QToolButton:focus, QPushButton:focus, QComboBox:focus {{
                    background: {button_hover};
                }}
                QToolButton:hover, QPushButton:hover, QComboBox:hover {{
                    background: {button_hover};
                }}
                QToolButton:disabled, QPushButton:disabled, QComboBox:disabled {{
                    background: {button_disabled};
                    color: #606060;
                }}
                QComboBox QAbstractItemView {{
                    background-color: palette(Base);
                }}
                QCheckBox::indicator {{
                    width: 16px;
                    height: 16px;
                    background-color: palette(Base);
                    border: 1px solid {editbox_frame};
                    border-radius: 4px;
                }}
                QCheckBox::indicator:hover {{
                    border-color: {checkbox_hover};
                }}
                QCheckBox::indicator:checked {{
                    image: url(resources/icons/{checkbox_glyph});
                    background-color: {checkbox};
                    border: 1px solid {checkbox};
                }}
                QCheckBox::indicator:checked:hover {{
                    background-color: {checkbox_hover};
                    border: 1px solid {checkbox_hover};
                }}
                QCheckBox::indicator:disabled {{
                    background-color: {disabled};
                    border-color: {button_disabled};
                }}
                QCheckBox::indicator:checked:disabled {{
                    background-color: {disabled};
                    border-color: {button_disabled};
                }}
            """
        app.setStyleSheet(style_sheet.format(**colors))

    # Apply current color scheme
    current_scheme = app.styleHints().colorScheme()
    apply_palette(current_scheme)

    # Signal for live updates
    def on_color_scheme_changed(new_scheme):
        apply_palette(new_scheme)

    app.styleHints().colorSchemeChanged.connect(on_color_scheme_changed)

    return app
