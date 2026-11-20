from math import ceil

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtGui import QTextLayout, QTextOption
from PySide6.QtWidgets import QPlainTextEdit, QSizePolicy, QStyle, QStyleOptionSizeGrip, QStylePainter, QWidget


class _TextBoxResizeHandle(QWidget):
    def __init__(self, editor):
        super().__init__(editor)
        self.editor = editor
        self._drag_start = None
        self.setFixedSize(16, 16)
        self.setCursor(Qt.CursorShape.SizeVerCursor)
        self.setToolTip("Drag to resize the prompt")

    def paintEvent(self, event):
        option = QStyleOptionSizeGrip()
        option.initFrom(self)
        option.corner = Qt.Corner.BottomRightCorner
        painter = QStylePainter(self)
        painter.drawControl(QStyle.ControlElement.CE_SizeGrip, option)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = (event.globalPosition().y(), self.editor.height())
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_start is not None:
            start_y, start_height = self._drag_start
            height = round(start_height + event.globalPosition().y() - start_y)
            self.editor.setFixedHeight(max(self.editor.default_height(), height))
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = None
            event.accept()
        else:
            super().mouseReleaseEvent(event)


class QtTextBox(QPlainTextEdit):
    """A line-sized plain text editor with an optional vertical resize handle."""

    editingFinished = Signal()

    def __init__(self, parent, *, lines: int, resizable: bool = False):
        super().__init__(parent)
        self._lines = lines
        self._resize_handle = _TextBoxResizeHandle(self) if resizable else None
        self.setTabChangesFocus(True)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        if self._resize_handle is not None:
            self.setViewportMargins(0, 0, 0, self._resize_handle.height())
            self.viewport().installEventFilter(self)
        self.ensurePolished()
        self.setFixedHeight(self.default_height())

    def default_height(self):
        margins = self.contentsMargins()
        handle_height = self._resize_handle.height() if self._resize_handle is not None else 0
        # Qt needs one extra pixel below the last line to avoid scrolling it.
        return ceil(self._lines * self.fontMetrics().lineSpacing() + 2 * self.document().documentMargin()
                    + margins.top() + margins.bottom() + handle_height) + 1

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() in (QEvent.Type.FontChange, QEvent.Type.StyleChange) and hasattr(self, "_lines"):
            height = self.default_height()
            self.setFixedHeight(max(height, self.height()) if self._resize_handle is not None else height)

    def eventFilter(self, watched, event):
        if watched == self.viewport() and event.type() == QEvent.Type.Resize and self._resize_handle is not None:
            # Stay beside the scrollbar when it appears, without covering its arrows.
            viewport = self.viewport().geometry()
            self._resize_handle.move(viewport.right() + 1 - self._resize_handle.width(), viewport.bottom() + 1)
        return super().eventFilter(watched, event)

    def focusOutEvent(self, event):
        super().focusOutEvent(event)
        self.editingFinished.emit()


class QtTextPreview(QtTextBox):
    """Display the beginning of a prompt without editing or scrolling its text."""

    clicked = Signal()

    def __init__(self, parent, *, lines: int = 2, clickable: bool = False):
        self._full_text = ""
        self._clickable = clickable
        self._pressed = False
        super().__init__(parent, lines=lines)
        self.setProperty("textPreview", True)
        self.setReadOnly(True)
        self.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        if clickable:
            self.viewport().setCursor(Qt.CursorShape.PointingHandCursor)

    def setText(self, text: str):
        self._full_text = text
        self.setAccessibleDescription(text)
        self._update_preview()

    def _update_preview(self):
        width = max(1, self.viewport().width() - 2 * self.document().documentMargin() - 1)
        # QTextLayout uses a line separator for explicit breaks within a paragraph.
        text = self._full_text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\u2028")
        utf16_text = text.encode("utf-16-le")
        layout = QTextLayout(text, self.font())
        option = QTextOption()
        option.setWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        layout.setTextOption(option)
        layout.beginLayout()
        lines = []
        for _ in range(self._lines):
            line = layout.createLine()
            if not line.isValid():
                break
            line.setLineWidth(width)
            # Qt counts UTF-16 code units, including two units for emoji.
            start = 2 * line.textStart()
            end = start + 2 * line.textLength()
            lines.append(utf16_text[start:end].decode("utf-16-le").rstrip())
        truncated = layout.createLine().isValid()
        layout.endLayout()
        if truncated:
            lines[-1] = self.fontMetrics().elidedText(lines[-1] + "…", Qt.TextElideMode.ElideRight, int(width))
        self.setPlainText("\n".join(lines))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_preview()

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() in (QEvent.Type.FontChange, QEvent.Type.StyleChange) and hasattr(self, "_lines"):
            self._update_preview()

    def mousePressEvent(self, event):
        if self._clickable and self.isEnabled() and event.button() == Qt.MouseButton.LeftButton:
            self._pressed = True
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if self._pressed and event.button() == Qt.MouseButton.LeftButton:
            self._pressed = False
            if self.isEnabled() and self.viewport().rect().contains(event.position().toPoint()):
                self.clicked.emit()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        # Let the surrounding sample list scroll instead.
        event.ignore()
