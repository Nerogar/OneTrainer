from abc import ABCMeta

from PySide6.QtWidgets import QWidget


class QtABCMeta(type(QWidget), ABCMeta):
    """Combined metaclass that resolves the conflict between Qt's Shiboken metaclass and ABCMeta."""
