import os
from math import ceil

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QUICK_BACKEND", "software")

from modules.ui.GeneralTabController import GeneralTabController
from modules.ui.GeneralTabView import GeneralTabView
from modules.util.config.TrainConfig import TrainConfig
from modules.util.ui.PySide6UIState import PySide6UIState
from modules.util.ui.theme import ThemeController
from modules.util.ui.validation import flush_and_validate_all

import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QPoint, QSettings, Qt, QUrl
from PySide6.QtGui import QAccessible, QColor, QHelpEvent, QPalette
from PySide6.QtQml import QQmlComponent
from PySide6.QtQuickWidgets import QQuickWidget
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QCheckBox, QLabel, QLineEdit, QToolButton, QToolTip
from shiboken6 import isValid


@pytest.fixture(scope="module")
def app():
    instance = QApplication.instance() or QApplication([])
    instance.setStyle("Fusion")
    return instance


@pytest.fixture
def make_view(app, monkeypatch, tmp_path):
    theme = ThemeController(app, QSettings(str(tmp_path / "theme.ini"), QSettings.Format.IniFormat))
    monkeypatch.setattr("modules.ui.GeneralTabView.get_theme_controller", lambda: theme)
    views = []

    def make(count, *, cloud=False, enabled=False):
        config = TrainConfig.default_values()
        config.cloud.enabled = cloud
        config.multi_gpu = enabled
        state = PySide6UIState(config)
        controller = GeneralTabController(
            local_gpu_count=count,
            on_workspace_dir_change=lambda *_: None,
            on_always_on_tensorboard_toggle=lambda: None,
        )
        view = GeneralTabView(None, controller, state)
        views.append(view)
        return view, config, state

    yield make
    for view in views:
        if isValid(view):
            view.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    theme.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_main_panel_inputs_share_a_flexible_label_column(make_view, app):
    view, _, _ = make_view(2)
    workspace = next(entry for entry in view.findChildren(QLineEdit)
                     if entry._validator.var_name == "workspace_dir")
    panel = workspace.parentWidget().parentWidget()
    fields = panel.findChildren(QLineEdit) + panel.findChildren(QCheckBox)
    assert len(fields) == 10
    label = next(label for label in panel.findChildren(QLabel) if label.text() == "Debug mode")
    original_text = label.text()
    view.show()

    def input_x():
        QTest.qWait(20)  # Allow nested layout requests and window resizing to settle.
        positions = {field.mapTo(panel, QPoint(0, 0)).x() for field in fields}
        assert len(positions) == 1
        return positions.pop()

    for width in (1400, 950, 1200):
        view.resize(width, 700)
        before = input_x()
        label.setText("A much longer debugging label that needs additional space")
        assert input_x() > before
        label.setText(original_text)
        assert input_x() == before
    font = label.font()
    font.setPointSizeF(30)
    label.setFont(font)
    assert input_x() > before


@pytest.mark.parametrize("count", [0, 1, None])
def test_unavailable_local_gpu_disables_and_clears_enable(make_view, count):
    view, config, state = make_view(count, enabled=True)
    assert not config.multi_gpu
    assert not view.multi_gpu_switch.isChecked()
    assert not view.multi_gpu_switch.isEnabled()
    assert view.multi_gpu_details.isHidden()
    assert not view.multi_gpu_details.content.isEnabled()
    assert "detected" in view.multi_gpu_switch.toolTip() or "failed" in view.multi_gpu_switch.toolTip()
    assert not view.findChildren(QToolButton)
    assert config.async_gradient_reduce_buffer == 100
    state.get_var("multi_gpu").set(True)
    assert not config.multi_gpu


@pytest.mark.parametrize("count,cloud", [(2, False), (4, False), (0, True), (None, True)])
def test_available_switch_controls_details_and_preserves_values(make_view, count, cloud):
    view, config, _ = make_view(count, cloud=cloud, enabled=True)
    before = config.to_dict()
    assert not view.multi_gpu_details.isHidden()
    assert view.multi_gpu_switch.isEnabled()
    assert view.multi_gpu_switch.isChecked()
    view.multi_gpu_switch.click()
    assert view.multi_gpu_details.isHidden()
    assert not config.multi_gpu
    view.multi_gpu_switch.click()
    assert not view.multi_gpu_details.isHidden()
    assert config.to_dict() == before


def test_cloud_local_transitions_keep_advanced_settings(make_view):
    view, config, state = make_view(1, cloud=True, enabled=True)
    state.get_var("device_indexes").set("0,2")
    state.get_var("async_gradient_reduce_buffer").set("256")
    state.get_var("cloud.enabled").set(False)
    assert not config.multi_gpu
    assert view.multi_gpu_details.isHidden()
    assert not view.multi_gpu_switch.isEnabled()
    state.get_var("cloud.enabled").set(True)
    assert view.multi_gpu_switch.isEnabled()
    assert not config.multi_gpu
    assert config.device_indexes == "0,2"
    assert config.async_gradient_reduce_buffer == 256


def test_bulk_load_cloud_preset_is_not_cleared_mid_update(make_view, app):
    view, config, state = make_view(1)
    loaded = TrainConfig.default_values()
    loaded.multi_gpu = True
    loaded.cloud.enabled = True
    loaded.async_gradient_reduce_buffer = 256
    for _ in range(3):
        state.update(loaded)
        app.processEvents()
        assert config.multi_gpu
        assert view.multi_gpu_switch.isChecked()
        assert view.multi_gpu_switch.isEnabled()
    state.update(loaded)
    app.processEvents()
    assert not view.multi_gpu_details.isHidden()
    loaded.cloud.enabled = False
    state.update(loaded)
    view.refresh_availability()  # synchronous post-load hook
    assert not config.multi_gpu
    assert config.async_gradient_reduce_buffer == 256
    assert view.multi_gpu_details.isHidden()


def test_moved_validation_fields_and_collapsed_values(make_view):
    view, config, _ = make_view(2)
    fields = {entry._validator.var_name: entry for entry in view.findChildren(QLineEdit)}
    timesteps = fields["validation_timesteps"]
    timesteps.setText("100,499,900")
    assert timesteps._validator.flush() is None
    assert config.validation_timesteps == "100,499,900"
    timesteps.setText("1000")
    assert timesteps._validator.flush() is not None
    delta = fields["early_stopping_min_delta"]
    delta.setText("")
    assert delta._validator.flush() is None
    assert config.early_stopping_min_delta is None
    view.multi_gpu_switch.click()
    buffer = fields["async_gradient_reduce_buffer"]
    buffer.setText("256")
    view.multi_gpu_switch.click()
    assert buffer._validator.flush() is None
    assert config.async_gradient_reduce_buffer == 256
    buffer.setText("invalid")
    assert buffer._validator.flush() is not None


@pytest.mark.parametrize(("section", "field", "invalid"), [
    ("validation", "validation_timesteps", "1000"),
    ("multi_gpu", "async_gradient_reduce_buffer", "-1"),
])
def test_training_preflight_skips_disabled_sections(make_view, section, field, invalid):
    view, _, state = make_view(2)
    state.get_var(section).set(True)
    entry = next(entry for entry in view.findChildren(QLineEdit) if entry._validator.var_name == field)
    entry.setText(invalid)
    errors = flush_and_validate_all()
    assert len(errors) == 1
    assert errors[0].startswith(entry._validator.display_name)

    state.get_var(section).set(False)
    assert flush_and_validate_all() == []
    state.get_var(section).set(True)
    assert flush_and_validate_all() == errors


@pytest.mark.parametrize("name", ["multi_gpu", "validation"])
def test_disabled_categories_show_only_enable_switch(make_view, name):
    view, config, _ = make_view(2)
    switch = getattr(view, name + "_switch")
    details = getattr(view, name + "_details")
    assert details.isHidden()
    switch.click()
    assert getattr(config, name)
    assert not details.isHidden()
    switch.click()
    assert not getattr(config, name)
    assert details.isHidden()


@pytest.mark.parametrize("name", ["multi_gpu", "validation"])
def test_animation_and_rapid_reversal(make_view, app, name):
    view, config, _ = make_view(2)
    view.resize(1100, 600)
    view.show()
    app.processEvents()
    switch = getattr(view, name + "_switch")
    details = getattr(view, name + "_details")
    panel = details.parentWidget().parentWidget()
    collapsed_height = panel.height()
    switch.click()
    assert details.is_animating
    QTest.qWait(70)
    assert 0 < details.height() < details.content.sizeHint().height()
    switch.click()
    QTest.qWait(40)
    switch.click()
    QTest.qWait(240)
    assert getattr(config, name)
    assert not details.isHidden()
    assert details.height() >= details.content.sizeHint().height()
    assert panel.height() > collapsed_height
    switch.click()
    assert not details.content.isEnabled()
    QTest.qWait(240)
    assert details.isHidden()
    assert not details.is_animating


def test_disabled_gpu_tooltip_is_shown_on_hover(make_view, app):
    view, _, _ = make_view(1)
    view.show()
    app.processEvents()
    switch = view.multi_gpu_switch
    event = QHelpEvent(QEvent.Type.ToolTip, QPoint(1, 1), switch.mapToGlobal(QPoint(1, 1)))
    QApplication.sendEvent(switch, event)
    assert QToolTip.text() == switch.toolTip()
    assert "1 detected" in QToolTip.text()
    assert not any("detected" in label.text() for label in view.findChildren(QLabel))
    QToolTip.hideText()


def test_preset_sync_updates_validation_without_animation(make_view, app):
    view, config, state = make_view(2)
    view.show()
    loaded = TrainConfig.default_values()
    loaded.validation = True
    loaded.validation_timesteps = "100,499,900"
    state.update(loaded)
    app.processEvents()
    assert config.validation
    assert view.validation_switch.isChecked()
    assert not view.validation_details.isHidden()
    assert not view.validation_details.is_animating
    loaded.validation = False
    state.update(loaded)
    app.processEvents()
    assert view.validation_details.isHidden()
    assert config.validation_timesteps == "100,499,900"


def test_destroying_view_during_deferred_preset_refresh_removes_traces(make_view, app, capsys):
    view, _, state = make_view(1)
    view.show()
    loaded = TrainConfig.default_values()
    loaded.cloud.enabled = True
    loaded.multi_gpu = True
    loaded.validation = True
    state.update(loaded)
    view.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert not isValid(view)
    state.get_var("cloud.enabled").set(False)
    state.get_var("validation").set(False)
    state.get_var("multi_gpu").set(False)
    app.processEvents()
    assert "RuntimeError" not in capsys.readouterr().err


def test_surface_colors_follow_palette_without_changing_app_style(make_view, app):
    view, _, _ = make_view(1)
    original = app.palette()
    stylesheet = app.styleSheet()
    try:
        dark = QPalette(original)
        dark.setColor(QPalette.ColorRole.Window, QColor("#202020"))
        dark.setColor(QPalette.ColorRole.WindowText, QColor("#eeeeee"))
        app.setPalette(dark)
        assert "#2e2e2e" in view.validation_panel.styleSheet()
        assert view.validation_switch.quickWindow().color().name() == "#2e2e2e"
        light = QPalette(original)
        light.setColor(QPalette.ColorRole.Window, QColor("#efefef"))
        light.setColor(QPalette.ColorRole.WindowText, QColor("#202020"))
        app.setPalette(light)
        assert "#f9f9f9" in view.validation_panel.styleSheet()
        assert view.validation_switch.quickWindow().color().name() == "#f9f9f9"
        assert app.styleSheet() == stylesheet
    finally:
        app.setPalette(original)


def test_appearance_moves_between_columns_without_losing_theme(make_view, app):
    view, _, state = make_view(2)
    view.show()
    panel = view.appearance_panel
    dropdown = view.theme_dropdown
    dropdown.setCurrentIndex(dropdown.findData("dark"))
    assert panel.parentWidget() is view.validation_column
    for enabled in (True, False, True):
        state.get_var("validation").set(enabled)
        app.processEvents()
        expected = view.device_column if enabled else view.validation_column
        assert panel.parentWidget() is expected
        assert expected.layout().itemAtPosition(1, 0).widget() is panel
        assert view.theme_dropdown is dropdown
        assert dropdown.currentData() == "dark"
        assert panel.isVisible()
    loaded = TrainConfig.default_values()
    state.update(loaded)
    app.processEvents()
    assert panel.parentWidget() is view.validation_column


@pytest.mark.parametrize("name", ["multi_gpu", "validation"])
def test_qml_switch_mouse_keyboard_and_state_binding(make_view, app, name):
    view, config, state = make_view(2)
    view.show()
    app.processEvents()
    switch = getattr(view, name + "_switch")
    assert isinstance(switch, QQuickWidget)
    assert switch.status() == QQuickWidget.Status.Ready
    indicator = switch.rootObject().property("indicator")
    assert indicator.y() >= 0
    assert indicator.y() + indicator.height() <= switch.height()
    assert not switch.isChecked()
    QTest.mouseClick(switch, Qt.MouseButton.LeftButton, pos=switch.rect().center())
    assert getattr(config, name)
    assert switch.isChecked()
    switch.setFocus()
    QTest.keyClick(switch, Qt.Key.Key_Space)
    assert not getattr(config, name)
    assert not switch.isChecked()
    state.get_var(name).set(True)
    assert switch.isChecked()


def test_disabled_qml_switch_cannot_enable_multi_gpu(make_view, app):
    view, config, _ = make_view(1)
    view.show()
    app.processEvents()
    switch = view.multi_gpu_switch
    assert not switch.rootObject().property("enabled")
    QTest.mouseClick(switch, Qt.MouseButton.LeftButton, pos=switch.rect().center())
    QTest.keyClick(switch, Qt.Key.Key_Space)
    switch.click()
    assert not switch.isChecked()
    assert not config.multi_gpu


def test_switch_uses_default_control_dimensions(make_view):
    view, _, _ = make_view(2)
    switch = view.validation_switch
    component = QQmlComponent(switch.engine())
    component.setData(b"import QtQuick.Controls\nSwitch {}", QUrl())
    default = component.create()
    assert default is not None
    try:
        assert switch.width() == ceil(default.implicitWidth())
        assert switch.height() == ceil(default.implicitHeight())
        for name in ("padding", "spacing"):
            assert switch.rootObject().property(name) == default.property(name)
    finally:
        default.deleteLater()


def test_switch_loads_module_without_external_qml_file(make_view, monkeypatch):
    def reject_file_source(*_args):
        raise AssertionError("Switch must load from Qt's module, not an external QML file")

    monkeypatch.setattr(QQuickWidget, "setSource", reject_file_source)
    view, _, _ = make_view(2)
    for switch in (view.validation_switch, view.multi_gpu_switch):
        assert switch.status() == QQuickWidget.Status.Ready
        root = switch.rootObject()
        assert root.property("focus")
        interface = QAccessible.queryAccessibleInterface(root)
        assert interface is not None
        assert interface.text(QAccessible.Text.Name) == switch.accessibleName()


def test_switch_framebuffer_corners_match_panel(make_view, app):
    view, _, _ = make_view(1)
    view.show()
    app.processEvents()
    for switch in (view.validation_switch, view.multi_gpu_switch):
        image = switch.grabFramebuffer()
        assert not image.isNull()
        assert image.pixelColor(0, 0) == switch.quickWindow().color()
        assert image.pixelColor(image.width() - 1, image.height() - 1) == switch.quickWindow().color()


def test_validation_and_appearance_bottoms_align_in_both_states(make_view, app):
    view, _, state = make_view(2)
    view.show()

    def bottom(widget):
        return widget.mapTo(view, QPoint(0, widget.height())).y()

    device_panel = view.device_column.layout().itemAtPosition(0, 0).widget()

    def assert_collapsed_alignment():
        assert view.appearance_panel.parentWidget() is view.validation_column
        assert abs(bottom(view.appearance_panel) - bottom(device_panel)) <= 1
        for panel in (view.validation_panel, view.appearance_panel):
            assert panel.height() > panel.sizeHint().height()

    QTest.qWait(20)
    assert_collapsed_alignment()
    for width in (1200, 950, 1400):
        view.resize(width, 700)
        state.get_var("validation").set(True)
        QTest.qWait(240)
        assert abs(bottom(view.validation_panel) - bottom(view.appearance_panel)) <= 1
        state.get_var("validation").set(False)
        QTest.qWait(240)
        assert_collapsed_alignment()
    loaded = TrainConfig.default_values()
    loaded.validation = True
    state.update(loaded)
    QTest.qWait(20)  # Allow the deferred preset refresh and layout pass to finish.
    assert abs(bottom(view.validation_panel) - bottom(view.appearance_panel)) <= 1
    loaded.validation = False
    state.update(loaded)
    QTest.qWait(20)
    assert_collapsed_alignment()
