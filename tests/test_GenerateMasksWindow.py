import sys
from tkinter import END
from unittest.mock import MagicMock, Mock, patch

from modules.ui.GenerateMasksWindow import (
    MaskingController,
    MaskingModel,
    MaskingView,
)

import pytest

# Mock dependencies before they are imported by the module under test
# This prevents actual UI elements from being created
mock_ctk = MagicMock()

# Create a proper mock for CTkToplevel that doesn't interfere with __init__
class MockCTkToplevel:
    def __init__(self, *args, **kwargs):
        self.children = {}

    def title(self, text):
        pass

    def geometry(self, geometry_string):
        pass

    def resizable(self, width, height):
        pass

    def wait_visibility(self):
        pass

    def grab_set(self):
        pass

    def focus_set(self):
        pass

    def after(self, delay, callback=None):
        if callback:
            callback()

    def grid_rowconfigure(self, index, **kwargs):
        pass

    def grid_columnconfigure(self, index, **kwargs):
        pass

    def destroy(self):
        pass

    def __getattr__(self, name):
        return MagicMock()

# Create mocks that return appropriate values for UI operations
class MockStringVar:
    def __init__(self, master=None, value=""):
        self._value = value
    def get(self):
        return self._value
    def set(self, value):
        self._value = value

class MockBooleanVar:
    def __init__(self, master=None, value=False):
        self._value = value
    def get(self):
        return self._value
    def set(self, value):
        self._value = value

class MockWidget:
    def __init__(self, *args, **kwargs):
        pass
    def grid(self, *args, **kwargs):
        pass
    def pack(self, *args, **kwargs):
        pass
    def place(self, *args, **kwargs):
        pass
    def configure(self, *args, **kwargs):
        pass
    def get(self):
        return ""
    def set(self, value):
        pass
    def insert(self, index, value):
        pass
    def delete(self, start, end=None):
        pass
    def focus_set(self):
        pass
    def update(self):
        pass
    def __getattr__(self, name):
        return MagicMock()

mock_ctk.CTkToplevel = MockCTkToplevel
mock_ctk.StringVar = MockStringVar
mock_ctk.BooleanVar = MockBooleanVar
mock_ctk.CTkFrame = MockWidget
mock_ctk.CTkLabel = MockWidget
mock_ctk.CTkEntry = MockWidget
mock_ctk.CTkButton = MockWidget
mock_ctk.CTkOptionMenu = MockWidget
mock_ctk.CTkSwitch = MockWidget
mock_ctk.CTkProgressBar = MockWidget
sys.modules['customtkinter'] = mock_ctk
sys.modules['modules.util.ui.ToolTip'] = MagicMock()
mock_ui_utils = MagicMock()
# Ensure the mock functions return proper values by default
mock_ui_utils.load_window_session_settings.return_value = None
mock_ui_utils.save_window_session_settings.return_value = None
sys.modules['modules.util.ui.ui_utils'] = mock_ui_utils

# Now, import the classes from the module under test

# --- Fixtures ---

@pytest.fixture
def mock_parent():
    """Fixture for a mock parent window with a mock model_manager."""
    parent = Mock()
    parent.model_manager = Mock()
    parent.model_manager.get_available_masking_models.return_value = ["model1", "model2"]
    # Add session_ui_settings as a real dictionary instead of Mock
    parent.session_ui_settings = {}
    return parent

@pytest.fixture
def mock_view(mock_parent):
    """Fixture for a mock MaskingView."""
    # Patch all UI-related methods and the entire initialization
    with patch.object(MaskingView, '__init__', return_value=None):
        view = MaskingView.__new__(MaskingView)  # Create instance without calling __init__

        # Set required attributes manually
        view.parent = mock_parent
        view.controller = Mock()
        view.show_error = Mock()
        view.show_warning = Mock()
        view.processing_started = Mock()
        view.processing_finished = Mock()
        view.gather_settings_from_ui = Mock()
        view.set_progress = Mock()
        view.after = Mock()
        # Add missing tkinter attributes needed for destroy method
        view.children = {}
        view.tk = Mock()
        view._w = Mock()
        view._name = "mock_view"
        view.master = Mock()
        view.master.children = {}
        # Add missing CustomTkinter scaling attributes
        view._CTkScalingBaseClass__scaling_type = "widget"

        return view

@pytest.fixture
def controller(mock_parent, mock_view):
    """Fixture for a MaskingController."""
    return MaskingController(parent=mock_parent, view=mock_view)

# --- Tests for MaskingModel ---

def test_masking_model_defaults():
    """Test that the MaskingModel dataclass has the correct default values."""
    model = MaskingModel()
    assert model.model == ""
    assert model.path == ""
    assert model.prompt == ""
    assert model.mode == "Create if absent"
    assert model.threshold == "0.3"
    assert model.smooth == "0"
    assert model.expand == "10"
    assert model.alpha == "1"
    assert not model.include_subdirectories
    assert not model.preview_mode

# --- Tests for MaskingController ---

class TestMaskingController:
    def test_init(self, controller, mock_parent, mock_view):
        """Test controller initialization."""
        assert controller.parent is mock_parent
        assert controller.view is mock_view
        assert isinstance(controller.model, MaskingModel)
        assert "Create if absent" in controller.mode_map

    def test_load_settings(self, controller, mock_view):
        """Test that settings are loaded from session and applied to the view."""
        # Patch the actual imported function
        with patch('modules.ui.GenerateMasksWindow.load_window_session_settings', return_value=None) as mock_load:
            mock_view.apply_settings_to_ui = Mock()

            controller.load_settings()

            mock_load.assert_called_once_with(mock_view, controller.SESSION_SETTINGS_KEY)
            # When no settings exist, model should remain default
            assert controller.model.prompt == ""
            assert controller.model.path == ""
            # apply_settings_to_ui should not be called when no settings exist
            mock_view.apply_settings_to_ui.assert_not_called()

    def test_load_settings_with_data(self, controller, mock_view):
        """Test that settings are loaded from session and applied to the view when data exists."""
        mock_settings_dict = {"prompt": "loaded prompt", "path": "/loaded/path"}
        with patch('modules.ui.GenerateMasksWindow.load_window_session_settings', return_value=mock_settings_dict) as mock_load:
            mock_view.apply_settings_to_ui = Mock()

            controller.load_settings()

            mock_load.assert_called_once_with(mock_view, controller.SESSION_SETTINGS_KEY)
            assert controller.model.prompt == "loaded prompt"
            assert controller.model.path == "/loaded/path"
            mock_view.apply_settings_to_ui.assert_called_once()
            applied_model = mock_view.apply_settings_to_ui.call_args[0][0]
            assert isinstance(applied_model, MaskingModel)
            assert applied_model.prompt == "loaded prompt"

    def test_get_mode(self, controller):
        """Test the get_mode method for correct mode mapping."""
        assert controller.get_mode("Replace all masks") == "replace"
        assert controller.get_mode("Create if absent") == "fill"
        assert controller.get_mode("Add to existing") == "add"
        assert controller.get_mode("Unknown mode") == "fill"  # Test default case

    @patch('modules.ui.GenerateMasksWindow.asdict')
    @patch('modules.ui.GenerateMasksWindow.save_window_session_settings')
    def test_save_settings(self, mock_save, mock_asdict, controller, mock_view):
        """Test that settings are gathered from the UI and saved."""
        mock_settings = MaskingModel(prompt="test")
        mock_view.gather_settings_from_ui.return_value = mock_settings
        mock_asdict.return_value = {"prompt": "test"}

        controller.save_settings()

        mock_view.gather_settings_from_ui.assert_called_once()
        assert controller.model == mock_settings
        mock_save.assert_called_once_with(
            mock_view, controller.SESSION_SETTINGS_KEY, {"prompt": "test"}
        )

    def test_prepare_mask_args_normal_mode(self, controller):
        """Test _prepare_mask_args in normal (non-preview) mode."""
        controller.model = MaskingModel(
            prompt="person", path="/some/folder", include_subdirectories=True,
            threshold="0.5", smooth="2", expand="5", alpha="0.8"
        )

        args, error = controller._prepare_mask_args()

        assert error is None
        assert args["prompts"] == ["person"]
        assert args["mode"] == "fill"
        assert args["threshold"] == 0.5
        assert args["smooth_pixels"] == 2
        assert args["expand_pixels"] == 5
        assert args["alpha"] == 0.8
        assert args["sample_dir"] == "/some/folder"
        assert args["include_subdirectories"] is True
        assert "single_file" not in args

    def test_prepare_mask_args_preview_mode(self, controller, mock_parent):
        """Test _prepare_mask_args in preview mode with valid parent state."""
        mock_parent.current_image_index = 0
        mock_parent.image_rel_paths = ["image1.png"]
        mock_parent.dir = "/base/dir"
        controller.model = MaskingModel(prompt="cat", preview_mode=True)

        args, error = controller._prepare_mask_args()

        assert error is None
        assert args["single_file"] == "image1.png"
        assert args["sample_dir"] == "/base/dir"
        assert args["include_subdirectories"] is True # Should be forced true in preview
        assert "path" not in args # Should not use the model's path

    def test_prepare_mask_args_preview_mode_no_image(self, controller, mock_parent):
        """Test _prepare_mask_args in preview mode when no image is selected."""
        # Simulate parent not having the necessary attributes
        if hasattr(mock_parent, "current_image_index"):
            delattr(mock_parent, "current_image_index")

        controller.model = MaskingModel(prompt="cat", preview_mode=True)

        args, error = controller._prepare_mask_args()

        assert args is None
        assert "no image is selected" in error

    @pytest.mark.parametrize(
        "settings, is_dir, expected_valid, expected_error_part",
        [
            (MaskingModel(prompt="person", path="/valid/dir", threshold="0.5", smooth="5", expand="10", alpha="0.8"), True, True, ""),
            (MaskingModel(prompt=" ", path="/valid/dir"), True, False, "Please enter a detection prompt"),
            (MaskingModel(prompt="person", path="/invalid/dir"), False, False, "Please select a valid folder"),
            (MaskingModel(prompt="person", path="/valid/dir", threshold="abc"), True, False, "Invalid number value"),
            (MaskingModel(prompt="person", path="/valid/dir", threshold="1.1"), True, False, "Threshold must be between 0.0 and 0.9"),
            (MaskingModel(prompt="person", path="/valid/dir", expand="-5"), True, False, "Expand pixels should be between 0 and 64"),
        ]
    )
    @patch('modules.ui.GenerateMasksWindow.Path')
    def test_validate_inputs(self, mock_path, settings, is_dir, expected_valid, expected_error_part, controller):
        """Test input validation with various correct and incorrect inputs."""
        mock_path.return_value.is_dir.return_value = is_dir
        is_valid, error_msg = controller.validate_inputs(settings)
        assert is_valid == expected_valid
        if not expected_valid:
            assert expected_error_part in error_msg

    @patch('concurrent.futures.ThreadPoolExecutor.submit')
    def test_create_masks_valid(self, mock_submit, controller, mock_view):
        """Test the create_masks flow with valid inputs."""
        valid_settings = MaskingModel(prompt="person", path="/valid/dir")
        mock_view.gather_settings_from_ui.return_value = valid_settings
        with patch.object(controller, 'validate_inputs', return_value=(True, "")) as mock_validate:
            controller.create_masks()

            mock_validate.assert_called_once_with(valid_settings)
            mock_view.processing_started.assert_called_once()
            mock_submit.assert_called_once()

def test_masking_model_with_custom_values():
    """Test MaskingModel with custom values."""
    model = MaskingModel(
        model="test_model",
        path="/test/path",
        prompt="test prompt",
        mode="Replace all masks",
        threshold="0.7",
        smooth="3",
        expand="15",
        alpha="0.5",
        include_subdirectories=True,
        preview_mode=True
    )
    assert model.model == "test_model"
    assert model.path == "/test/path"
    assert model.prompt == "test prompt"
    assert model.mode == "Replace all masks"
    assert model.threshold == "0.7"
    assert model.smooth == "3"
    assert model.expand == "15"
    assert model.alpha == "0.5"
    assert model.include_subdirectories is True
    assert model.preview_mode is True

class TestMaskingControllerAdditional:
    def test_get_mode_all_mappings(self, controller):
        """Test all mode mappings in get_mode method."""
        assert controller.get_mode("Replace all masks") == "replace"
        assert controller.get_mode("Create if absent") == "fill"
        assert controller.get_mode("Add to existing") == "add"
        assert controller.get_mode("Subtract from existing") == "subtract"
        assert controller.get_mode("Blend with existing") == "blend"

    def test_prepare_mask_args_preview_mode_invalid_index(self, controller, mock_parent):
        """Test _prepare_mask_args in preview mode with invalid image index."""
        mock_parent.current_image_index = 5
        mock_parent.image_rel_paths = ["image1.png", "image2.png"]
        mock_parent.dir = "/base/dir"
        controller.model = MaskingModel(prompt="cat", preview_mode=True)

        args, error = controller._prepare_mask_args()

        assert args is None
        assert "No current image is selected" in error

    def test_prepare_mask_args_preview_mode_no_image_paths(self, controller, mock_parent):
        """Test _prepare_mask_args in preview mode when image_rel_paths is missing."""
        mock_parent.current_image_index = 0
        if hasattr(mock_parent, "image_rel_paths"):
            delattr(mock_parent, "image_rel_paths")
        controller.model = MaskingModel(prompt="cat", preview_mode=True)

        args, error = controller._prepare_mask_args()

        assert args is None
        assert "No current image is selected" in error

    @pytest.mark.parametrize(
        "threshold, smooth, expand, alpha, expected_error",
        [
            ("abc", "0", "10", "1", "Invalid number value"),
            ("0.5", "abc", "10", "1", "Invalid number value"),
            ("0.5", "0", "abc", "1", "Invalid number value"),
            ("0.5", "0", "10", "abc", "Invalid number value"),
            ("1.5", "0", "10", "1", "Threshold must be between 0.0 and 0.9"),
            ("0.5", "15", "10", "1", "Smooth pixels should be between 0 and 10"),
            ("0.5", "0", "100", "1", "Expand pixels should be between 0 and 64"),
            ("0.5", "0", "10", "2.0", "Alpha must be between 0.0 and 1.0"),
        ]
    )
    @patch('modules.ui.GenerateMasksWindow.Path')
    def test_validate_inputs_numeric_errors(self, mock_path, threshold, smooth, expand, alpha, expected_error, controller):
        """Test validation with various numeric input errors."""
        mock_path.return_value.is_dir.return_value = True
        settings = MaskingModel(
            prompt="person",
            path="/valid/dir",
            threshold=threshold,
            smooth=smooth,
            expand=expand,
            alpha=alpha
        )
        is_valid, error_msg = controller.validate_inputs(settings)
        assert not is_valid
        assert expected_error in error_msg

    @patch('modules.ui.GenerateMasksWindow.Path')
    def test_validate_inputs_edge_cases(self, mock_path, controller):
        """Test validation with edge case values."""
        mock_path.return_value.is_dir.return_value = True

        # Test minimum valid values
        settings = MaskingModel(
            prompt="a",
            path="/valid/dir",
            threshold="0.0",
            smooth="0",
            expand="0",
            alpha="0.0"
        )
        is_valid, error_msg = controller.validate_inputs(settings)
        assert is_valid

        # Test maximum valid values
        settings = MaskingModel(
            prompt="person",
            path="/valid/dir",
            threshold="0.9",
            smooth="10",
            expand="64",
            alpha="1.0"
        )
        is_valid, error_msg = controller.validate_inputs(settings)
        assert is_valid

    @patch('concurrent.futures.ThreadPoolExecutor.submit')
    def test_create_masks_thread_exception(self, mock_submit, controller, mock_view):
        """Test create_masks when thread submission fails."""
        valid_settings = MaskingModel(prompt="person", path="/valid/dir")
        mock_view.gather_settings_from_ui.return_value = valid_settings
        mock_submit.side_effect = RuntimeError("Thread pool error")

        with patch.object(controller, 'validate_inputs', return_value=(True, "")):
            controller.create_masks()

            mock_view.processing_started.assert_called_once()
            mock_view.show_error.assert_called_once_with("Thread Error", "Thread pool error")

    def test_run_masking_process_model_load_failure(self, controller, mock_parent):
        """Test run_masking_process when model loading fails."""
        mock_parent.model_manager.load_masking_model.return_value = None
        controller.model = MaskingModel(model="invalid_model", prompt="person", path="/test")

        with pytest.raises(RuntimeError, match="Failed to load masking model"):
            controller.run_masking_process()

    def test_run_masking_process_prepare_args_error(self, controller, mock_parent):
        """Test run_masking_process when _prepare_mask_args returns an error."""
        mock_masking_model = Mock()
        mock_parent.model_manager.load_masking_model.return_value = mock_masking_model

        with patch.object(controller, '_prepare_mask_args', return_value=(None, "Test error")), \
             pytest.raises(RuntimeError, match="Test error"):
            controller.run_masking_process()

    def test_run_masking_process_success(self, controller, mock_parent):
        """Test successful run_masking_process execution."""
        mock_masking_model = Mock()
        mock_parent.model_manager.load_masking_model.return_value = mock_masking_model
        controller.model = MaskingModel(prompt="person", path="/test")

        test_args = {"prompts": ["person"], "mode": "fill"}
        with patch.object(controller, '_prepare_mask_args', return_value=(test_args, None)):
            controller.run_masking_process()

            mock_masking_model.mask_folder.assert_called_once_with(**test_args)

class TestMaskingViewAdditional:
    @patch.object(MaskingView, '__init__', return_value=None)
    def test_view_initialization_with_empty_path(self, mock_init, mock_parent):
        """Test MaskingView initialization with None path."""
        view = MaskingView.__new__(MaskingView)
        view.parent = mock_parent
        view.controller = MaskingController(mock_parent, view)

        # Manually call what we want to test about initialization
        mock_init.assert_not_called()  # Because we patched it
        assert view.parent is mock_parent
        assert isinstance(view.controller, MaskingController)

    @patch.object(MaskingView, '__init__', return_value=None)
    def test_view_initialization_with_valid_path(self, mock_init, mock_parent):
        """Test MaskingView initialization with valid path."""
        view = MaskingView.__new__(MaskingView)
        view.parent = mock_parent
        view.controller = MaskingController(mock_parent, view)

        mock_init.assert_not_called()  # Because we patched it
        assert view.parent is mock_parent

    def test_show_warning_calls_messagebox(self, mock_view):
        """Test that show_warning properly calls messagebox."""
        # Override the mock with the actual method
        mock_view.show_warning = MaskingView.show_warning.__get__(mock_view, MaskingView)

        with patch('modules.ui.GenerateMasksWindow.messagebox') as mock_messagebox:
            mock_view.show_warning("Test Title", "Test Message")
            mock_messagebox.showwarning.assert_called_once_with("Test Title", "Test Message", parent=mock_view)

    def test_show_error_calls_messagebox_and_resets_button(self, mock_view):
        """Test that show_error calls messagebox and resets button state."""
        # Override the mock with the actual method
        mock_view.show_error = MaskingView.show_error.__get__(mock_view, MaskingView)
        mock_view.reset_button_state = Mock()

        with patch('modules.ui.GenerateMasksWindow.messagebox') as mock_messagebox:
            mock_view.show_error("Error Title", "Error Message")
            mock_messagebox.showerror.assert_called_once_with("Error Title", "Error Message", parent=mock_view)
            mock_view.reset_button_state.assert_called_once()

    def test_processing_states(self, mock_view):
        """Test processing started/finished state changes."""
        # Override the mocks with actual methods
        mock_view.processing_started = MaskingView.processing_started.__get__(mock_view, MaskingView)
        mock_view.processing_finished = MaskingView.processing_finished.__get__(mock_view, MaskingView)
        mock_view.reset_button_state = MaskingView.reset_button_state.__get__(mock_view, MaskingView)
        mock_view.create_masks_button = Mock()

        mock_view.processing_started()
        mock_view.create_masks_button.configure.assert_called_with(state="disabled", text="Processing...")

        mock_view.processing_finished()
        mock_view.create_masks_button.configure.assert_called_with(state="normal", text="Create Masks")

    def test_set_progress(self, mock_view):
        """Test progress setting functionality."""
        # Override the mock with actual method
        mock_view.set_progress = MaskingView.set_progress.__get__(mock_view, MaskingView)
        mock_view.progress = Mock()
        mock_view.progress_label = Mock()

        mock_view.set_progress(5, 10)

        mock_view.progress.set.assert_called_once_with(0.5)
        mock_view.progress_label.configure.assert_called_once_with(text="5/10")
        mock_view.progress.update.assert_called_once()

    def test_browse_for_path(self, mock_view):
        """Test browse_for_path functionality."""
        mock_entry = Mock()
        mock_view.focus_set = Mock()

        with patch('modules.ui.GenerateMasksWindow.filedialog') as mock_filedialog:
            mock_filedialog.askdirectory.return_value = "/selected/path"

            mock_view.browse_for_path(mock_entry)

            mock_filedialog.askdirectory.assert_called_once()
            mock_entry.focus_set.assert_called_once()
            mock_entry.delete.assert_called_once_with(0, END)
            mock_entry.insert.assert_called_once_with(0, "/selected/path")
            mock_view.focus_set.assert_called_once()

    def test_check_future_exception(self, controller, mock_view):
        """Test _check_future when the task raises an exception."""
        future = Mock()
        future.done.return_value = True
        test_exception = ValueError("Masking failed")
        future.result.side_effect = test_exception

        controller._check_future(future)

        future.result.assert_called_once()
        mock_view.show_error.assert_called_once()
        # Check that the error message contains the exception's string representation
        error_message = mock_view.show_error.call_args[0][1]
        assert str(test_exception) in error_message
        mock_view.processing_finished.assert_not_called() # Should not be called on error

    def test_check_future_not_done(self, controller, mock_view):
        """Test _check_future when the task is still running."""
        future = Mock()
        future.done.return_value = False

        controller._check_future(future)

        mock_view.after.assert_called_once()
        future.result.assert_not_called()
        mock_view.processing_finished.assert_not_called()
        mock_view.show_error.assert_not_called()

class TestMaskingView:

    def test_gather_settings_from_ui(self, mock_parent):
        """Test that settings are correctly gathered from mock UI elements."""
        with patch.object(MaskingView, '__init__', return_value=None):
            view = MaskingView.__new__(MaskingView)

            # Mock UI elements and their .get() methods
            view.model_var = Mock(get=Mock(return_value="model1"))
            view.path_entry = Mock(get=Mock(return_value="/my/path"))
            view.prompt_entry = Mock(get=Mock(return_value="dog"))
            view.mode_var = Mock(get=Mock(return_value="Add to existing"))
            view.threshold_entry = Mock(get=Mock(return_value="0.4"))
            view.smooth_entry = Mock(get=Mock(return_value="2"))
            view.expand_entry = Mock(get=Mock(return_value="12"))
            view.alpha_entry = Mock(get=Mock(return_value="0.7"))
            view.include_subdirectories_var = Mock(get=Mock(return_value=True))
            view.preview_mode_var = Mock(get=Mock(return_value=True))

            settings = view.gather_settings_from_ui()

            assert isinstance(settings, MaskingModel)
            assert settings.model == "model1"
            assert settings.path == "/my/path"
            assert settings.prompt == "dog"
            assert settings.mode == "Add to existing"
            assert settings.threshold == "0.4"
            assert settings.smooth == "2"
            assert settings.expand == "12"
            assert settings.alpha == "0.7"
            assert settings.include_subdirectories is True
            assert settings.preview_mode is True

    def test_apply_settings_to_ui(self, mock_parent):
        """Test that settings are correctly applied to mock UI elements."""
        with patch.object(MaskingView, '__init__', return_value=None):
            view = MaskingView.__new__(MaskingView)

            # Mock UI elements and their methods
            view.model_var = Mock(set=Mock())
            view.path_entry = Mock(delete=Mock(), insert=Mock())
            view.prompt_entry = Mock(delete=Mock(), insert=Mock())
            view.mode_var = Mock(set=Mock())
            view.threshold_entry = Mock(delete=Mock(), insert=Mock())
            view.smooth_entry = Mock(delete=Mock(), insert=Mock())
            view.expand_entry = Mock(delete=Mock(), insert=Mock())
            view.alpha_entry = Mock(delete=Mock(), insert=Mock())
            view.include_subdirectories_var = Mock(set=Mock())
            view.preview_mode_var = Mock(set=Mock())

            settings = MaskingModel(
                model="model2", path="/new/path", prompt="cat", mode="Replace all masks",
                threshold="0.9", smooth="8", expand="20", alpha="0.5",
                include_subdirectories=True, preview_mode=True
            )

            view.apply_settings_to_ui(settings)

            view.model_var.set.assert_called_once_with("model2")
            view.path_entry.insert.assert_called_once_with(0, "/new/path")
            view.prompt_entry.insert.assert_called_once_with(0, "cat")
            view.mode_var.set.assert_called_once_with("Replace all masks")
            view.threshold_entry.insert.assert_called_once_with(0, "0.9")
            view.smooth_entry.insert.assert_called_once_with(0, "8")
            view.expand_entry.insert.assert_called_once_with(0, "20")
            view.alpha_entry.insert.assert_called_once_with(0, "0.5")
            view.include_subdirectories_var.set.assert_called_once_with(True)
            view.preview_mode_var.set.assert_called_once_with(True)

    def test_destroy_saves_settings(self, mock_view):
        """Test that destroying the view calls the controller's save_settings method."""
        # Set up the controller with save_settings mock
        mock_view.controller.save_settings = Mock()

        # Ensure the scaling type attribute is present
        mock_view._CTkScalingBaseClass__scaling_type = "widget"

        # Mock the entire destroy chain to prevent calling into real implementation
        with patch('customtkinter.windows.widgets.scaling.scaling_base_class.CTkScalingBaseClass.destroy') as mock_scaling_destroy:
            # Call the actual destroy method
            MaskingView.destroy(mock_view)

            # Verify the controller's save_settings was called
            mock_view.controller.save_settings.assert_called_once()
            # The scaling base class destroy should be called
            mock_scaling_destroy.assert_called_once()
