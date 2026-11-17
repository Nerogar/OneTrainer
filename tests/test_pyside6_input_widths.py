import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from modules.util.config.TrainConfig import TrainConfig
from modules.util.ui import pyside6_components as components
from modules.util.ui.pyside6_util import fit_form, limit_input_width
from modules.util.ui.PySide6UIState import PySide6UIState

import pytest
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication, QComboBox, QLineEdit, QPushButton, QWidget


@pytest.fixture
def form():
    app = QApplication.instance() or QApplication([])
    parent = QWidget()
    components._layout(parent).setColumnStretch(0, 1)
    config = TrainConfig.default_values()
    state = PySide6UIState(config)
    yield app, parent, config, state
    parent.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def build_input(kind, parent, state, **kwargs):
    if kind == "entry":
        return components.entry(parent, 0, 0, state, "tensorboard_port", **kwargs)
    if kind == "path_entry":
        return components.path_entry(parent, 0, 0, state, "workspace_dir", mode="dir", **kwargs)
    if kind == "time_entry":
        return components.time_entry(parent, 0, 0, state, "validate_after", "validate_after_unit", **kwargs)
    return components.options(parent, 0, 0, ["cuda", "cpu"], state, "train_device", **kwargs)


@pytest.mark.parametrize("kind", ["entry", "path_entry", "time_entry", "options"])
def test_optional_caps_survive_layout_and_resize(form, kind):
    app, parent, _, state = form
    widget = build_input(kind, parent, state, max_width=40)
    parent.show()
    app.processEvents()
    text_widget = widget
    extra_width = 0
    if kind in ("path_entry", "time_entry"):
        text_widget = widget.findChild(QLineEdit)
    if kind == "path_entry":
        extra_width = widget.findChild(QPushButton).width() + widget.layout().horizontalSpacing()
    elif kind == "time_entry":
        extra_width = widget.findChild(QComboBox).sizeHint().width() + widget.layout().horizontalSpacing()
    expected_cap = text_widget.fontMetrics().horizontalAdvance("0" * 40) + 12 + extra_width
    for width in (600, 1000, 700):
        parent.resize(width, 250)
        app.processEvents()
        assert widget.maximumWidth() == expected_cap
        assert widget.width() <= expected_cap
        assert widget.x() == 0


def test_entry_cap_below_default_minimum_keeps_validation_and_binding(form):
    app, parent, config, state = form
    entry = components.entry(parent, 0, 0, state, "tensorboard_port", max_width=4)
    parent.resize(600, 100)
    parent.show()
    app.processEvents()
    assert entry.width() == entry.fontMetrics().horizontalAdvance("0000") + 12
    assert entry.width() < 140
    entry.setText("7000")
    assert entry._validator.flush() is None
    assert config.tensorboard_port == 7000
    entry.setText("invalid")
    assert entry._validator.flush() is not None
    state.get_var("tensorboard_port").set("8000")
    assert entry.text() == "8000"


def test_omitting_cap_preserves_entry_sizing(form):
    app, parent, _, state = form
    entry = components.entry(parent, 0, 0, state, "tensorboard_port")
    assert entry.minimumWidth() == 140
    parent.resize(600, 100)
    parent.show()
    app.processEvents()
    assert entry.width() > 320


@pytest.mark.parametrize("cap", [4, 20, 40, None])
def test_path_gap_and_total_control_cap(form, cap):
    app, parent, _, state = form
    path = components.path_entry(parent, 0, 0, state, "workspace_dir", max_width=cap)
    entry = path.findChild(QLineEdit)
    button = path.findChild(QPushButton)
    parent.resize(600, 100)
    parent.show()
    app.processEvents()
    assert button.x() - (entry.x() + entry.width()) == 8
    if cap is not None:
        text_cap = entry.fontMetrics().horizontalAdvance("0" * cap) + 12
        assert entry.maximumWidth() == text_cap
        assert path.maximumWidth() == text_cap + button.width() + 8
        assert path.width() <= path.maximumWidth()
    state.get_var("workspace_dir").set("workspace/new-run")
    assert entry.text() == "workspace/new-run"


def test_fit_form_respects_explicit_combo_cap(form):
    app, parent, config, state = form
    combo = components.options(parent, 0, 0, ["cuda", "cpu"], state, "train_device", max_width=4)
    cap = combo.maximumWidth()
    fit_form(parent)
    parent.show()
    app.processEvents()
    assert combo.width() <= cap
    combo.setCurrentText("cpu")
    assert config.train_device == "cpu"


@pytest.mark.parametrize("kind", ["entry", "path_entry"])
def test_character_caps_use_control_font_and_do_not_limit_text(form, kind):
    app, parent, _, state = form
    # Use a string field so a long value is valid and remains editable.
    small = components.path_entry(parent, 0, 0, state, "workspace_dir", max_width=20) if kind == "path_entry" else (
        components.entry(parent, 0, 0, state, "workspace_dir", max_width=20)
    )
    font = parent.font()
    font.setPointSizeF(20)
    large_parent = QWidget(parent)
    large_parent.setFont(font)
    components._layout(parent).addWidget(large_parent, 1, 0)
    large = components.path_entry(large_parent, 0, 0, state, "cache_dir", max_width=20) if kind == "path_entry" else (
        components.entry(large_parent, 0, 0, state, "cache_dir", max_width=20)
    )
    parent.show()
    app.processEvents()
    assert large.maximumWidth() > small.maximumWidth()
    for widget in (small, large):
        entry = widget.findChild(QLineEdit) if kind == "path_entry" else widget
        assert entry.maximumWidth() == entry.fontMetrics().horizontalAdvance("0" * 20) + 12
        long_value = "long-directory-name/" * 10
        entry.setText(long_value)
        assert entry.text() == long_value


def test_max_width_matches_existing_character_width_utility(form):
    _, parent, _, state = form
    direct = components.entry(parent, 0, 0, state, "tensorboard_port", max_width=4)
    manual = components.entry(parent, 1, 0, state, "tensorboard_port")
    limit_input_width(manual, characters=4)
    assert direct.minimumWidth() == manual.minimumWidth()
    assert direct.maximumWidth() == manual.maximumWidth()
