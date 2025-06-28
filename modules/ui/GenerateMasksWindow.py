import concurrent.futures
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from tkinter import END, filedialog, messagebox
from typing import Any

from modules.util.ui.ToolTip import ToolTip
from modules.util.ui.ui_utils import (
    load_window_session_settings,
    save_window_session_settings,
    set_window_icon,
)

import customtkinter as ctk

logger = logging.getLogger(__name__)

@dataclass(slots=True)
class MaskingModel:
    model: str = ""
    path: str = ""
    prompt: str = ""
    mode: str = "Create if absent"
    threshold: str = "0.3"
    smooth: str = "0"
    expand: str = "10"
    alpha: str = "1"
    include_subdirectories: bool = False
    preview_mode: bool = False

class MaskingController:
    SESSION_SETTINGS_KEY = "generate_masks_window_settings"
    _executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

    def __init__(self, parent: Any, view: 'MaskingView'):
        self.parent = parent
        self.view = view
        self.model = MaskingModel()
        self.mode_map = {
            "Replace all masks": "replace",
            "Create if absent": "fill",
            "Add to existing": "add",
            "Subtract from existing": "subtract",
            "Blend with existing": "blend",
        }

    def load_settings(self):
        settings_dict = load_window_session_settings(self.view, self.SESSION_SETTINGS_KEY)
        if settings_dict:
            self.model = MaskingModel(**settings_dict)
            self.view.apply_settings_to_ui(self.model)

    def save_settings(self):
        self.model = self.view.gather_settings_from_ui()
        save_window_session_settings(self.view, self.SESSION_SETTINGS_KEY, asdict(self.model))

    def get_mode(self, mode_str: str) -> str:
        return self.mode_map.get(mode_str, "fill")

    def _prepare_mask_args(self) -> tuple[dict[str, Any] | None, str | None]:
        """Prepare arguments for the masking model based on UI settings."""
        mode = self.get_mode(self.model.mode)

        args = {
            "prompts": [self.model.prompt],
            "mode": mode,
            "alpha": float(self.model.alpha),
            "threshold": float(self.model.threshold),
            "smooth_pixels": int(self.model.smooth),
            "expand_pixels": int(self.model.expand),
            "progress_callback": self.view.set_progress,
        }

        if self.model.preview_mode:
            if not hasattr(self.parent, "current_image_index"):
                return None, "Preview mode is enabled, but no image is selected."

            if not (hasattr(self.parent, "image_rel_paths") and
                    0 <= self.parent.current_image_index < len(self.parent.image_rel_paths)):
                return None, "No current image is selected for preview."

            current_image_rel_path = self.parent.image_rel_paths[self.parent.current_image_index]
            logger.info("Preview mode: Processing only image %s", current_image_rel_path)

            args.update({
                "sample_dir": self.parent.dir,
                "include_subdirectories": True,
                "single_file": current_image_rel_path
            })
        else:
            args.update({
                "sample_dir": self.model.path,
                "include_subdirectories": self.model.include_subdirectories
            })

        return args, None

    def run_masking_process(self):
        masking_model = self.parent.model_manager.load_masking_model(
            self.model.model
        )

        if masking_model is None:
            raise RuntimeError(f"Failed to load masking model: {self.model.model}")

        if hasattr(masking_model, "model_manager"):
            masking_model.model_manager = self.parent.model_manager

        mask_args, error = self._prepare_mask_args()

        if error:
            raise RuntimeError(error)

        if mask_args:
            masking_model.mask_folder(**mask_args)

    def create_masks(self):
        self.model = self.view.gather_settings_from_ui()
        is_valid, error = self.validate_inputs(self.model)
        if not is_valid:
            self.view.show_warning("Invalid Input", error)
            return

        self.view.processing_started()
        try:
            future = self._executor.submit(self.run_masking_process)
            self.view.after(100, lambda: self._check_future(future))
        except Exception as e:
            logger.exception("Failed to start masking thread")
            self.view.show_error("Thread Error", str(e))

    def _check_future(self, future):
        if not future.done():
            self.view.after(100, lambda: self._check_future(future))
            return

        try:
            future.result()
            self.view.processing_finished()
            if hasattr(self.parent, "image_handler") and hasattr(
                self.parent.image_handler, "load_image_data"
            ):
                self.parent.image_handler.load_image_data()
                self.parent.refresh_ui()
        except Exception as e:
            logger.exception("Error in masking process")
            self.view.show_error("Processing Error", f"An error occurred during mask creation:\n{str(e)}")

    def validate_inputs(self, settings: MaskingModel) -> tuple[bool, str]:
        """Validate all user inputs. Returns (is_valid, error_message)."""
        validators = [
            ("Threshold", settings.threshold, float, 0.0, 0.9, "Threshold must be between 0.0 and 0.9 for usable results"),
            ("Smooth", settings.smooth, int, 0, 10, "Smooth pixels should be between 0 and 10"),
            ("Expand", settings.expand, int, 0, 64, "Expand pixels should be between 0 and 64"),
            ("Alpha", settings.alpha, float, 0.0, 1.0, "Alpha must be between 0.0 and 1.0"),
        ]

        try:
            for _name, value_str, type_conv, min_val, max_val, msg in validators:
                value = type_conv(value_str)
                if not min_val <= value <= max_val:
                    return False, msg
        except ValueError:
            return False, "Invalid number value. Please check your inputs."

        if not settings.prompt.strip():
            return False, "Please enter a detection prompt"
        if not Path(settings.path).is_dir():
            return False, "Please select a valid folder"

        return True, ""

class MaskingView(ctk.CTkToplevel):
    __slots__ = (
        'parent', 'controller', 'models', 'model_var', 'modes', 'mode_var', 'preview_mode_var',
        'frame', 'model_dropdown', 'path_entry', 'path_button', 'prompt_entry', 'mode_dropdown',
        'threshold_entry', 'smooth_entry', 'expand_entry', 'alpha_entry', 'include_subdirectories_var',
        'progress_label', 'progress', 'create_masks_button'
    )

    def __init__(
            self, parent: Any, path: str | None, include_subdirectories: bool, *args: Any, **kwargs: Any
        ):
            """
            Window for generating masks for a folder of images

            Parameters:
                parent (`Tk`): the parent window
                path (`str`): the path to the folder
                include_subdirectories (`bool`): whether to include subdirectories
            """
            super().__init__(parent, *args, **kwargs)
            self.parent = parent
            self.controller = MaskingController(parent, self)

            if path is None:
                path = ""

            # Setup window properties
            self._setup_window("Batch generate masks", "360x480")

            # Get available models dynamically
            self.models = self.parent.model_manager.get_available_masking_models()
            self.model_var = ctk.StringVar(
                self, self.models[0] if self.models else ""
            )

            # Define modes with mapping to API values
            self.modes = list(self.controller.mode_map.keys())
            self.mode_var = ctk.StringVar(self, "Create if absent")

            # Add preview mode toggle
            self.preview_mode_var = ctk.BooleanVar(self, False)

            # Set up the UI
            self._create_layout(path, include_subdirectories)
            self.controller.load_settings() # Load settings after UI is created

            # If a non-empty path is passed from the parent, ensure it overrides any session-loaded path.
            # The `if path:` check handles None, empty strings, and other "falsy" values.
            if path:
                self.path_entry.delete(0, END)
                self.path_entry.insert(0, path)

    def apply_settings_to_ui(self, settings: MaskingModel):
        self.model_var.set(settings.model)
        self.path_entry.delete(0, END)
        self.path_entry.insert(0, settings.path)
        self.prompt_entry.delete(0, END)
        self.prompt_entry.insert(0, settings.prompt)
        self.mode_var.set(settings.mode)
        self.threshold_entry.delete(0, END)
        self.threshold_entry.insert(0, settings.threshold)
        self.smooth_entry.delete(0, END)
        self.smooth_entry.insert(0, settings.smooth)
        self.expand_entry.delete(0, END)
        self.expand_entry.insert(0, settings.expand)
        self.alpha_entry.delete(0, END)
        self.alpha_entry.insert(0, settings.alpha)
        self.include_subdirectories_var.set(settings.include_subdirectories)
        self.preview_mode_var.set(settings.preview_mode)


    def gather_settings_from_ui(self) -> MaskingModel:
        return MaskingModel(
            model=self.model_var.get(),
            path=self.path_entry.get(),
            prompt=self.prompt_entry.get(),
            mode=self.mode_var.get(),
            threshold=self.threshold_entry.get(),
            smooth=self.smooth_entry.get(),
            expand=self.expand_entry.get(),
            alpha=self.alpha_entry.get(),
            include_subdirectories=self.include_subdirectories_var.get(),
            preview_mode=self.preview_mode_var.get(),
        )

    def destroy(self):
        self.controller.save_settings()
        super().destroy()

    def _add_labeled_entry(self, row, label, default="", width=200, placeholder=None, tooltip=None):
        """
        Adds a labeled entry field to the UI at the specified row.

        Returns:
            ctk.CTkEntry: The created entry widget.
        """
        lbl = ctk.CTkLabel(self.frame, text=label, width=100)
        lbl.grid(row=row, column=0, sticky="w", padx=5, pady=5)
        entry = ctk.CTkEntry(self.frame, width=width, placeholder_text=placeholder)
        entry.insert(0, default)
        entry.grid(row=row, column=1, sticky="w", padx=5, pady=5)
        if tooltip:
            ToolTip(entry, tooltip)
        return entry

    def _setup_window(self, title, geometry):
        self.title(title)
        self.geometry(geometry)
        self.resizable(True, True)
        self.wait_visibility()
        self.grab_set()
        self.focus_set()
        self.after(200, lambda: set_window_icon(self))

    def _create_layout(self, path, include_subdirectories):
        # Create frame
        self.frame = ctk.CTkFrame(self, width=600, height=340)
        self.frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        # Configure the root window grid to allow frame expansion
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Create UI components
        self._create_model_selection()
        self._create_path_selection(path)
        self._create_mask_options()
        self._create_subdirectory_option(include_subdirectories)
        self._create_preview_option()
        self._create_progress_indicators()
        self._create_action_buttons()

    def _create_model_selection(self):
        model_label = ctk.CTkLabel(self.frame, text="Model", width=100)
        model_label.grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.model_dropdown = ctk.CTkOptionMenu(self.frame, variable=self.model_var, values=self.models, dynamic_resizing=False, width=200)
        self.model_dropdown.grid(row=0, column=1, sticky="w", padx=5, pady=5)

    def _create_path_selection(self, path):
        path_label = ctk.CTkLabel(self.frame, text="Folder", width=100)
        path_label.grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.path_entry = ctk.CTkEntry(self.frame, width=150)
        self.path_entry.insert(0, path)
        self.path_entry.grid(row=1, column=1, sticky="w", padx=5, pady=5)
        self.path_button = ctk.CTkButton(self.frame, width=30, text="...", command=lambda: self.browse_for_path(self.path_entry))
        self.path_button.grid(row=1, column=1, sticky="e", padx=5, pady=5)

    def _create_mask_options(self):
        # Prompt
        self.prompt_entry = self._add_labeled_entry(
            row=2,
            label="Prompt",
            default="",
            tooltip="Enter object to detect (e.g. 'person', 'dog', 'car')"
        )

        # Mode (special case: dropdown)
        mode_label = ctk.CTkLabel(self.frame, text="Mode", width=100)
        mode_label.grid(row=3, column=0, sticky="w", padx=5, pady=5)
        self.mode_dropdown = ctk.CTkOptionMenu(
            self.frame,
            variable=self.mode_var,
            values=self.modes,
            dynamic_resizing=False,
            width=200
        )
        self.mode_dropdown.grid(row=3, column=1, sticky="w", padx=5, pady=5)

        # Threshold
        self.threshold_entry = self._add_labeled_entry(
            row=4,
            label="Threshold",
            default="0.3",
            placeholder="0.0 - 1.0",
            tooltip="Confidence threshold: Lower values detect more objects but may include incorrect regions"
        )

        # Smooth
        self.smooth_entry = self._add_labeled_entry(
            row=5,
            label="Smooth",
            default="0",
            placeholder="0-10",
            tooltip="Additional smoothing (0=use built-in smoothing, higher values for extra smoothing)"
        )

        # Expand
        self.expand_entry = self._add_labeled_entry(
            row=6,
            label="Expand",
            default="10",
            placeholder="0-20",
            tooltip="Expansion pixels: Expands mask boundaries outward"
        )

        # Alpha
        self.alpha_entry = self._add_labeled_entry(
            row=7,
            label="Alpha",
            default="1",
            placeholder="0.0 - 1.0",
            tooltip="Blending strength when combining with existing masks"
        )

    def _create_subdirectory_option(self, include_subdirectories):
        include_subdirectories_label = ctk.CTkLabel(self.frame, text="Include subdirs", width=100)
        include_subdirectories_label.grid(row=8, column=0, sticky="w", padx=5, pady=5)
        self.include_subdirectories_var = ctk.BooleanVar(self, include_subdirectories)
        include_subdirectories_switch = ctk.CTkSwitch(self.frame, text="", variable=self.include_subdirectories_var)
        include_subdirectories_switch.grid(row=8, column=1, sticky="w", padx=5, pady=5)

    def _create_preview_option(self):
        """Create a toggle for test mode (only process current image)"""
        preview_mode_label = ctk.CTkLabel(self.frame, text="Test Run", width=100)
        preview_mode_label.grid(row=9, column=0, sticky="w", padx=5, pady=5)
        preview_mode_switch = ctk.CTkSwitch(self.frame, text="", variable=self.preview_mode_var)
        preview_mode_switch.grid(row=9, column=1, sticky="w", padx=5, pady=5)
        ToolTip(preview_mode_switch, "Masks only the current image to more quickly get feedback")

    def _create_progress_indicators(self):
        self.progress_label = ctk.CTkLabel(self.frame, text="Progress: 0/0", width=100)
        self.progress_label.grid(row=10, column=0, sticky="w", padx=5, pady=5)
        self.progress = ctk.CTkProgressBar(self.frame, orientation="horizontal", mode="determinate", width=200)
        self.progress.grid(row=10, column=1, sticky="w", padx=5, pady=5)

    def _create_action_buttons(self):
        self.create_masks_button = ctk.CTkButton(self.frame, text="Create Masks", width=310, command=self.controller.create_masks)
        self.create_masks_button.grid(row=11, column=0, columnspan=2, sticky="w", padx=5, pady=5)

    def browse_for_path(self, entry_box):
        path = filedialog.askdirectory()
        entry_box.focus_set()
        entry_box.delete(0, END)
        entry_box.insert(0, path)
        self.focus_set()

    def show_warning(self, title, message):
        messagebox.showwarning(title, message, parent=self)

    def show_error(self, title, message):
        self.reset_button_state()
        messagebox.showerror(title, message, parent=self)

    def processing_started(self):
        self.create_masks_button.configure(state="disabled", text="Processing...")

    def processing_finished(self):
        self.reset_button_state()

    def reset_button_state(self):
        self.create_masks_button.configure(state="normal", text="Create Masks")

    def set_progress(self, value, max_value):
        progress = value / max_value
        self.progress.set(progress)
        self.progress_label.configure(text=f"{value}/{max_value}")
        self.progress.update()
