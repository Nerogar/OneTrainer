from unittest.mock import Mock

from modules.ui.GeneralTabController import GeneralTabController

import pytest


@pytest.mark.parametrize("count", [None, 0, 1, 2, 4])
@pytest.mark.parametrize("cloud", [False, True])
def test_gpu_availability_and_reason(count, cloud):
    controller = GeneralTabController(
        local_gpu_count=count,
        on_workspace_dir_change=Mock(),
        on_always_on_tensorboard_toggle=Mock(),
    )
    available, tooltip = controller.get_multi_gpu_availability(cloud)
    assert available == (cloud or count in (2, 4))
    if available:
        assert tooltip == "Enable multi-GPU training"
    elif count is None:
        assert "detection failed" in tooltip
    else:
        assert f"{count} detected" in tooltip


def test_actions_delegate_to_application_callbacks():
    workspace_changed = Mock()
    tensorboard_toggled = Mock()
    controller = GeneralTabController(
        local_gpu_count=2,
        on_workspace_dir_change=workspace_changed,
        on_always_on_tensorboard_toggle=tensorboard_toggled,
    )
    workspace_changed.assert_not_called()
    tensorboard_toggled.assert_not_called()
    controller.on_workspace_dir_change("workspace/new-run")
    controller.on_always_on_tensorboard_toggle()
    workspace_changed.assert_called_once_with("workspace/new-run")
    tensorboard_toggled.assert_called_once_with()
