from PySide6.QtWidgets import QDialog, QLabel, QPushButton, QVBoxLayout


class PySide6CaptionUIView(QDialog):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.setWindowTitle("Dataset Tool")
        lo = QVBoxLayout(self)
        lo.addWidget(QLabel("The dataset tool has not been ported to Qt6 yet.\nYou can still use it by launching the CustomTkinter UI: scripts/train_ui_ctk.py"))
        ok = QPushButton("OK")
        ok.clicked.connect(self.accept)
        lo.addWidget(ok)
