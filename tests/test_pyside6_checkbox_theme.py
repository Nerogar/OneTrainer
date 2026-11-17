import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from modules.ui.ConceptTabView import ConceptWidgetView
from modules.util.config.ConceptConfig import ConceptConfig
from modules.util.ui.theme import _BASE_STYLESHEET, _DARK_STYLESHEET, _LIGHT_STYLESHEET

import pytest
from PIL import Image
from PySide6.QtCore import QCoreApplication, QEvent, QRect, QSize, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPalette
from PySide6.QtWidgets import QApplication, QCheckBox, QStyle, QStyleOptionButton, QWidget


@pytest.fixture(params=["normal", "concept"])
def checkbox(request, monkeypatch):
    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    original_stylesheet = app.styleSheet()
    original_palette = app.palette()
    parent = QWidget()
    parent.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
    if request.param == "concept":
        monkeypatch.setattr(ConceptWidgetView, "_get_preview_image", lambda _: Image.new("RGB", (150, 150)))
        def noop(*args):
            pass
        card = ConceptWidgetView(
            parent, ConceptConfig.default_values(), 0, noop, noop, noop, noop,
            SimpleNamespace(randomize_seed=False),
        )
        widget = card.findChild(QCheckBox)
        size = 20
    else:
        widget = QCheckBox(parent)
        widget.resize(40, 24)
        size = 16

    # Reproduce the Windows dark checkbox palette even on the offscreen plugin.
    palette = widget.palette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#1e1e1e"))
    palette.setColor(QPalette.ColorRole.Midlight, QColor("#5a5a5a"))
    palette.setColor(QPalette.ColorRole.Mid, QColor("#282828"))
    palette.setColor(QPalette.ColorRole.Text, QColor("#eeeeee"))
    app.setPalette(palette)
    palette.setColor(QPalette.ColorGroup.Active, QPalette.ColorRole.Base, QColor("#0078d4"))
    palette.setColor(QPalette.ColorGroup.Inactive, QPalette.ColorRole.Base, QColor("#001a68"))
    widget.setPalette(palette)
    parent.show()
    app.processEvents()
    yield app, widget, size
    parent.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.setPalette(original_palette)
    app.setStyleSheet(original_stylesheet)


def test_indicator_fits_widget_without_resizing_when_toggled_or_rethemed(checkbox):
    app, widget, size = checkbox
    for stylesheet in (_DARK_STYLESHEET, _LIGHT_STYLESHEET, _DARK_STYLESHEET):
        app.setStyleSheet(_BASE_STYLESHEET + stylesheet)
        app.processEvents()
        rects = []
        for checked in (False, True, False):
            widget.setChecked(checked)
            option = QStyleOptionButton()
            widget.initStyleOption(option)
            rect = widget.style().subElementRect(QStyle.SubElement.SE_CheckBoxIndicator, option, widget)
            assert rect.size() == QSize(size, size)
            assert widget.rect().contains(rect), "Indicator border is clipped by the checkbox widget"
            rects.append(rect)
        assert rects[0] == rects[1] == rects[2]


def render_indicator(widget, group, state):
    option = QStyleOptionButton()
    widget.initStyleOption(option)
    option.rect = QRect(0, 0, 32, 32)
    option.state = state
    if group != QPalette.ColorGroup.Disabled:
        option.state |= QStyle.StateFlag.State_Enabled
    if group == QPalette.ColorGroup.Active:
        option.state |= QStyle.StateFlag.State_Active
    option.palette.setCurrentColorGroup(group)
    image = QImage(32, 32, QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    widget.style().drawPrimitive(QStyle.PrimitiveElement.PE_IndicatorCheckBox, option, painter, widget)
    painter.end()
    return image


@pytest.mark.parametrize("group", [QPalette.ColorGroup.Active, QPalette.ColorGroup.Inactive, QPalette.ColorGroup.Disabled])
def test_dark_unchecked_fill_is_neutral_and_checked_rendering_is_preserved(checkbox, group):
    app, widget, _ = checkbox
    app.setStyleSheet(_BASE_STYLESHEET)
    app.processEvents()
    native = {
        state: render_indicator(widget, group, state)
        for state in (QStyle.StateFlag.State_On, QStyle.StateFlag.State_NoChange)
    }
    app.setStyleSheet(_BASE_STYLESHEET + _DARK_STYLESHEET)
    app.processEvents()
    unchecked = render_indicator(widget, group, QStyle.StateFlag.State_Off)
    assert unchecked.pixelColor(16, 16) == widget.palette().color(group, QPalette.ColorRole.Window)
    for state, image in native.items():
        assert render_indicator(widget, group, state) == image
