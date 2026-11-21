from modules.util.ui.UIState import BaseUIState

from PySide6.QtCore import QEasingCurve, QEvent, QRectF, QSize, Qt, QVariantAnimation
from PySide6.QtGui import QColor, QPainter, QPalette, QPen
from PySide6.QtWidgets import QAbstractButton


def _blend(background: QColor, foreground: QColor, amount: float) -> QColor:
    return QColor(*(round(a + (b - a) * amount) for a, b in zip(
        background.getRgb()[:3], foreground.getRgb()[:3], strict=True,
    )))


class QtToggleSwitch(QAbstractButton):
    """Pure Qt Widgets pill toggle switch bound to BaseUIState."""

    _WIDTH = 42
    _HEIGHT = 24
    _MARGIN = 2.0
    _TRACK_W = 38.0
    _TRACK_H = 20.0
    _THUMB_D = 14.0
    _THUMB_PAD = 3.0

    def __init__(self, parent, ui_state: BaseUIState, name: str, label: str):
        super().__init__(parent)
        self.setCheckable(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAccessibleName(label)
        self.setObjectName(label)
        self.setFixedSize(self._WIDTH, self._HEIGHT)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        self._hovered = False
        self._syncing = False
        self._thumb_pos = 0.0

        self._animation = QVariantAnimation(self)
        self._animation.setDuration(140)
        self._animation.setEasingCurve(QEasingCurve.Type.InOutCubic)
        self._animation.valueChanged.connect(self._set_thumb_pos)

        var = ui_state.get_var(name)
        initial = bool(var.get())
        self.setChecked(initial)
        self._thumb_pos = 1.0 if initial else 0.0

        self.toggled.connect(self._on_toggled)
        # clicked fires on mouse, keyboard, and click() — not programmatic
        # setChecked() — so preset loading cannot feed back into ui_state.
        self.clicked.connect(lambda checked: var.set(bool(checked)))
        binding = var._bind_widget(self._on_var_changed)
        self.destroyed.connect(lambda: var._unbind_widget(binding))

    def sizeHint(self) -> QSize:
        return QSize(self._WIDTH, self._HEIGHT)

    def _set_thumb_pos(self, value: float) -> None:
        self._thumb_pos = float(value)
        self.update()

    def _on_var_changed(self, value) -> None:
        self._syncing = True
        try:
            self.setChecked(bool(value))
        finally:
            self._syncing = False

    def _on_toggled(self, checked: bool) -> None:
        target = 1.0 if checked else 0.0
        self._animation.stop()
        if self.isVisible() and not self._syncing:
            self._animation.setStartValue(self._thumb_pos)
            self._animation.setEndValue(target)
            self._animation.start()
        else:
            self._thumb_pos = target
            self.update()

    def enterEvent(self, event) -> None:
        self._hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        self._hovered = False
        self.update()
        super().leaveEvent(event)

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() == QEvent.Type.EnabledChange:
            self.setCursor(
                Qt.CursorShape.PointingHandCursor if self.isEnabled() else Qt.CursorShape.ArrowCursor
            )
            self.update()
        elif event.type() in (QEvent.Type.PaletteChange, QEvent.Type.StyleChange):
            self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        palette = self.palette()
        window = palette.color(QPalette.ColorRole.Window)
        text = palette.color(QPalette.ColorRole.WindowText)
        accent = palette.color(QPalette.ColorRole.Highlight)
        accent_text = palette.color(QPalette.ColorRole.HighlightedText)
        dark = window.lightnessF() < 0.5
        enabled = self.isEnabled()
        pos = self._thumb_pos

        if enabled:
            off_track = _blend(window, QColor("black"), 0.28) if dark else _blend(window, text, 0.14)
            on_track = accent
            track_fill = _blend(off_track, on_track, pos)
            if self._hovered:
                track_fill = _blend(track_fill, QColor("white"), 0.10 if dark else 0.08)

            off_border = text if self._hovered else _blend(window, text, 0.42)
            on_border = accent.darker(115)
            track_border = _blend(off_border, on_border, pos)

            off_thumb = _blend(window, text, 0.78) if dark else QColor("white")
            on_thumb = accent_text
            thumb_fill = _blend(off_thumb, on_thumb, pos)
            thumb_border = None if dark else _blend(window, text, 0.28 * (1.0 - pos))
        else:
            off_track = _blend(window, text, 0.06)
            on_track = _blend(window, accent, 0.35)
            track_fill = _blend(off_track, on_track, pos)
            track_border = _blend(window, text, 0.22)
            thumb_fill = _blend(window, text, 0.36)
            thumb_border = None

        track_rect = QRectF(self._MARGIN, self._MARGIN, self._TRACK_W, self._TRACK_H)
        radius = self._TRACK_H / 2.0

        if enabled and self.hasFocus():
            focus_color = QColor(accent)
            focus_color.setAlpha(170)
            painter.setPen(QPen(focus_color, 1.5))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(track_rect.adjusted(-1.25, -1.25, 1.25, 1.25), radius + 1.25, radius + 1.25)

        painter.setPen(QPen(track_border, 1.0))
        painter.setBrush(track_fill)
        painter.drawRoundedRect(track_rect, radius, radius)

        travel = self._TRACK_W - self._THUMB_D - 2.0 * self._THUMB_PAD
        thumb_x = self._MARGIN + self._THUMB_PAD + pos * travel
        thumb_y = self._MARGIN + self._THUMB_PAD
        thumb_rect = QRectF(thumb_x, thumb_y, self._THUMB_D, self._THUMB_D)

        painter.setPen(QPen(thumb_border, 1.0) if thumb_border is not None else Qt.PenStyle.NoPen)
        painter.setBrush(thumb_fill)
        painter.drawEllipse(thumb_rect)
