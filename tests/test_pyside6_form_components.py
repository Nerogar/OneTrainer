import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QUICK_BACKEND", "software")

from modules.util.config.TrainConfig import TrainConfig
from modules.util.ui import pyside6_components as components
from modules.util.ui.PySide6UIState import PySide6UIState
from modules.util.ui.theme import ThemeController, apply_theme

import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QPoint, QSettings, Qt
from PySide6.QtGui import QColor, QHelpEvent, QPalette
from PySide6.QtTest import QSignalSpy, QTest
from PySide6.QtWidgets import QApplication, QLabel, QToolTip, QWidget
from shiboken6 import isValid


@pytest.fixture
def form(tmp_path):
    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    theme = ThemeController(app, QSettings(str(tmp_path / "theme.ini"), QSettings.Format.IniFormat))
    parent = QWidget()
    config = TrainConfig.default_values()
    state = PySide6UIState(config)
    yield app, parent, config, state
    if isValid(parent):
        parent.deleteLater()
    theme.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_shared_spacing_and_heading_defaults_preserve_explicit_overrides(form):
    _, parent, _, _ = form
    layout = components._layout(parent)
    assert layout.horizontalSpacing() == 10
    assert layout.verticalSpacing() == 14
    layout.setSpacing(3)
    assert components._layout(parent).verticalSpacing() == 3
    body = components.form_section(parent, 0, 0, "Reusable section")
    panel = body.parentWidget()
    assert panel.property("formPanel")
    assert panel.layout().contentsMargins().left() == 10
    assert body.layout().verticalSpacing() == 14
    heading = panel.findChild(QLabel)
    field = components.label(body, 0, 0, "Wrapped field", underline=True, wraplength=100, tooltip="Help")
    assert heading.font().family() == field.font().family() == parent.font().family()
    assert field.font().pointSizeF() >= 10
    assert heading.font().pointSizeF() == field.font().pointSizeF() + 1
    assert heading.font().bold() and not field.font().bold()
    assert components.header_label(parent, 1, 0, "Next section", top_spacing=8).contentsMargins().top() == 8
    assert field.font().underline() and field.wordWrap()
    assert field.maximumWidth() == 100
    assert field.alignment() == Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft
    assert "Help" in field.toolTip()


@pytest.mark.parametrize("base,text,surface,field_color", [
    ("#202020", "#eeeeee", "#2e2e2e", "#d5d5d5"),
    ("#efefef", "#202020", "#f9f9f9", "#393939"),
])
def test_nested_panel_and_switch_colors_follow_application_palette(form, base, text, surface, field_color):
    app, parent, _, state = form
    original = app.palette()
    outer = components.section_frame(parent, 0)
    body = components.form_section(outer, 0, 0, "Nested")
    inner = body.parentWidget()
    heading = inner.findChild(QLabel)
    switch, field, _ = components.toggle_details(body, "Enable", state, "validation")
    parent.show()
    try:
        palette = QPalette(original)
        palette.setColor(QPalette.ColorRole.Window, QColor(base))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(text))
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, QColor("#808080"))
        app.setPalette(palette)
        app.processEvents()
        for panel in (outer, inner):
            assert panel.palette().color(QPalette.ColorRole.Window).name() == surface
        assert switch.quickWindow().color().name() == surface
        assert field.palette().color(QPalette.ColorRole.WindowText).name() == field_color
        assert heading.palette().color(QPalette.ColorRole.WindowText).name() == text
        field.setEnabled(False)
        assert field.palette().color(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText).name() == "#808080"
        image = switch.grabFramebuffer()
        assert not image.isNull()
        assert image.pixelColor(0, 0).name() == surface
    finally:
        app.setPalette(original)


@pytest.mark.parametrize("initially_dark", [False, True])
def test_label_colors_survive_theme_stylesheet_refresh(form, initially_dark):
    app, parent, _, _ = form
    original_palette = app.palette()
    original_stylesheet = app.styleSheet()

    def set_theme(dark):
        palette = QPalette(original_palette)
        palette.setColor(QPalette.ColorRole.Window, QColor("#202020" if dark else "#efefef"))
        palette.setColor(QPalette.ColorRole.WindowText, QColor("#eeeeee" if dark else "#202020"))
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, QColor("#808080"))
        app.setPalette(palette)
        # ThemeController refreshes the application stylesheet after the native
        # palette change. This repolishes widgets and can restore cached palettes.
        apply_theme(app)
        app.processEvents()

    try:
        set_theme(initially_dark)
        body = components.form_section(parent, 0, 0, "Heading")
        heading = body.parentWidget().findChild(QLabel)
        field = components.label(body, 0, 0, "Field")
        disabled = components.label(body, 1, 0, "Disabled")
        disabled.setEnabled(False)
        plain = components.label(parent, 1, 0, "Outside a panel")
        parent.show()
        for dark in (initially_dark, not initially_dark, initially_dark, not initially_dark):
            set_theme(dark)
            for group in (QPalette.ColorGroup.Active, QPalette.ColorGroup.Inactive):
                assert heading.palette().color(group, QPalette.ColorRole.WindowText).name() == (
                    "#eeeeee" if dark else "#202020"
                )
                for label in (field, plain):
                    assert label.palette().color(group, QPalette.ColorRole.WindowText).name() == (
                        "#d5d5d5" if dark else "#393939"
                    )
            assert disabled.palette().color(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText).name() == "#808080"
    finally:
        app.setPalette(original_palette)
        app.setStyleSheet(original_stylesheet)


def test_switch_tracks_surface_when_reparented(form):
    app, parent, _, state = form
    body = components.form_section(parent, 0, 0, "Panel")
    switch, _, _ = components.toggle_details(body, "Enable", state, "validation")
    plain = components.inline_frame(parent, 1, 0)
    palette = plain.palette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#123456"))
    plain.setPalette(palette)
    switch.setParent(plain)
    app.processEvents()
    assert switch.quickWindow().color().name() == "#123456"
    switch.setParent(body)
    assert switch.quickWindow().color() == body.parentWidget().palette().color(QPalette.ColorRole.Window)
    # A new parent subscribes to application palette updates after the switch.
    late_panel = components.section_frame(parent, 2)
    switch.setParent(late_panel)
    original = app.palette()
    try:
        palette = QPalette(original)
        palette.setColor(QPalette.ColorRole.Window, QColor("#101010"))
        app.setPalette(palette)
        app.processEvents()
        assert switch.quickWindow().color() == late_panel.palette().color(QPalette.ColorRole.Window)
    finally:
        app.setPalette(original)


def test_collapsible_form_reversal_and_immediate_sync_preserve_fields(form):
    app, parent, config, state = form
    details = components.CollapsibleForm(parent)
    components._layout(parent).addWidget(details, 0, 0)
    components.label(details.content, 0, 0, "Port")
    entry = components.entry(details.content, 0, 1, state, "tensorboard_port")
    # Like any hidden widget, the caller reserves its width in the outer layout.
    parent.setMinimumWidth(details.minimumSizeHint().width())
    parent.resize(500, 180)
    parent.show()
    app.processEvents()
    width = parent.width()
    finished = QSignalSpy(details.transition_finished)
    details.set_expanded(True)
    QTest.qWait(60)
    assert details.is_animating
    assert 0 < details.height() < details.content.sizeHint().height()
    details.set_expanded(False)
    assert not details.content.isEnabled()
    QTest.qWait(25)
    details.set_expanded(True)
    QTest.qWait(240)
    assert not details.is_animating and details.expanded
    assert finished.count() == 1
    entry.setText("7000")
    details.set_expanded(False)
    details.set_expanded(True, animate=False)
    assert not details.is_animating and details.content.isEnabled()
    details.set_expanded(False, animate=False)
    assert details.isHidden() and not details.content.isEnabled()
    assert entry._validator.flush() is None
    assert config.tensorboard_port == 7000
    state.get_var("tensorboard_port").set("8000")
    assert entry.text() == "8000"
    details.set_expanded(True, animate=False)
    app.processEvents()
    assert parent.width() == width
    assert details.height() >= details.content.sizeHint().height()


@pytest.mark.parametrize("target", ["summary", "label", "switch"])
def test_disabled_toggle_tooltips_work_without_general_tab(form, target):
    app, parent, _, state = form
    switch, field, _ = components.toggle_details(parent, "Enable", state, "validation")
    widgets = {"summary": switch.parentWidget(), "label": field, "switch": switch}
    for widget in widgets.values():
        widget.setToolTip("Unavailable for this test")
    switch.setEnabled(False)
    field.setEnabled(False)
    parent.show()
    app.processEvents()
    widget = widgets[target]
    event = QHelpEvent(QEvent.Type.ToolTip, QPoint(1, 1), widget.mapToGlobal(QPoint(1, 1)))
    QApplication.sendEvent(widget, event)
    assert QToolTip.text() == "Unavailable for this test"
    assert field.buddy() is switch
    QToolTip.hideText()


def test_destroying_components_removes_bindings_and_palette_subscriptions(form, capsys):
    app, parent, _, state = form
    body = components.form_section(parent, 0, 0, "Transient")
    switch, _, details = components.toggle_details(body, "Enable", state, "validation")
    components.entry(details.content, 0, 0, state, "tensorboard_port")
    parent.show()
    app.processEvents()
    details.set_expanded(True)
    parent.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert not isValid(switch)
    assert not isValid(details)
    state.get_var("validation").set(True)
    state.get_var("tensorboard_port").set("9000")
    app.paletteChanged.emit(app.palette())
    app.processEvents()
    assert "RuntimeError" not in capsys.readouterr().err
