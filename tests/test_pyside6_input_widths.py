import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from modules.util.config.TrainConfig import TrainConfig
from modules.util.ui import pyside6_components as components
from modules.util.ui.pyside6_util import fit_form
from modules.util.ui.PySide6UIState import PySide6UIState

import pytest
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication, QLineEdit, QPushButton, QWidget


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
    if kind == "layer_filter_entry":
        return components.layer_filter_entry(
            parent, 0, 0, state, "layer_filter_preset", "Layer Filter", "", {"full": []},
            "layer_filter", "", "layer_filter_regex", "", **kwargs,
        )
    if kind == "options":
        return components.options(parent, 0, 0, ["cuda", "cpu"], state, "train_device", **kwargs)
    if kind == "options_kv":
        return components.options_kv(parent, 0, 0, [("GPU", "cuda"), ("CPU", "cpu")], state, "train_device", **kwargs)
    if kind == "options_adv":
        return components.options_adv(parent, 0, 0, ["cuda", "cpu"], state, "train_device", **kwargs)[0]
    return components.switch(parent, 0, 0, state, "debug_mode", **kwargs)


@pytest.mark.parametrize("kind", ["entry", "path_entry", "time_entry", "layer_filter_entry",
                                 "options", "options_kv", "options_adv", "switch"])
def test_optional_caps_survive_layout_and_resize(form, kind):
    app, parent, _, state = form
    widget = build_input(kind, parent, state, max_width=320)
    if kind == "layer_filter_entry":
        state.get_var("layer_filter_preset").set("custom")
    parent.show()
    for width in (600, 1000, 700):
        parent.resize(width, 250)
        app.processEvents()
        assert widget.maximumWidth() == 320
        assert widget.width() <= 320
        assert widget.x() == 0


def test_entry_cap_below_default_minimum_keeps_validation_and_binding(form):
    app, parent, config, state = form
    entry = components.entry(parent, 0, 0, state, "tensorboard_port", max_width=80)
    parent.resize(600, 100)
    parent.show()
    app.processEvents()
    assert entry.width() == 80
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


@pytest.mark.parametrize("cap", [180, 320, None])
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
        assert path.width() <= cap
    state.get_var("workspace_dir").set("workspace/new-run")
    assert entry.text() == "workspace/new-run"


def test_fit_form_respects_explicit_combo_cap(form):
    app, parent, config, state = form
    combo = components.options(parent, 0, 0, ["cuda", "cpu"], state, "train_device", max_width=40)
    fit_form(parent)
    parent.show()
    app.processEvents()
    assert combo.width() <= 40
    combo.setCurrentText("cpu")
    assert config.train_device == "cpu"
