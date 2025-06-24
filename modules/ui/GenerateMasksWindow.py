import concurrent.futures
import logging
from dataclasses import dataclass
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

@dataclass
class MaskWindowSettings:
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

class GenerateMasksWindow(ctk.CTkToplevel):

    SESSION_SETTINGS_KEY = "generate_masks_window_settings"
    _executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

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
            self.modes = [
                "Replace all masks",
                "Create if absent",
                "Add to existing",
                "Subtract from existing",
                "Blend with existing",
            ]
            self.mode_var = ctk.StringVar(self, "Create if absent")

            # Add preview mode toggle
            self.preview_mode_var = ctk.BooleanVar(self, False)

            # Set up the UI
            self._create_layout(path, include_subdirectories)
            self._load_session_settings() # Load settings after UI is created

            # If a non-empty path is passed from the parent, ensure it overrides any session-loaded path.
            # The `if path:` check handles None, empty strings, and other "falsy" values.
            if path:
                self.path_entry.delete(0, END)
                self.path_entry.insert(0, path)

    def _apply_settings_to_ui(self, settings: MaskWindowSettings):
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


    def _gather_settings_from_ui(self) -> MaskWindowSettings:
        return MaskWindowSettings(
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

    def _load_session_settings(self):
        settings_dict = load_window_session_settings(self, self.SESSION_SETTINGS_KEY)
        if settings_dict:
            settings = MaskWindowSettings(**settings_dict)
            self._apply_settings_to_ui(settings)

    def _save_session_settings(self):
        settings = self._gather_settings_from_ui()
        save_window_session_settings(self, self.SESSION_SETTINGS_KEY, settings.__dict__)

    def destroy(self):
        self._save_session_settings()
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
        self.model_label = ctk.CTkLabel(self.frame, text="Model", width=100)
        self.model_label.grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.model_dropdown = ctk.CTkOptionMenu(self.frame, variable=self.model_var, values=self.models, dynamic_resizing=False, width=200)
        self.model_dropdown.grid(row=0, column=1, sticky="w", padx=5, pady=5)

    def _create_path_selection(self, path):
        self.path_label = ctk.CTkLabel(self.frame, text="Folder", width=100)
        self.path_label.grid(row=1, column=0, sticky="w", padx=5, pady=5)
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
        self.mode_label = ctk.CTkLabel(self.frame, text="Mode", width=100)
        self.mode_label.grid(row=3, column=0, sticky="w", padx=5, pady=5)
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
        self.include_subdirectories_label = ctk.CTkLabel(self.frame, text="Include subdirs", width=100)
        self.include_subdirectories_label.grid(row=8, column=0, sticky="w", padx=5, pady=5)
        self.include_subdirectories_var = ctk.BooleanVar(self, include_subdirectories)
        self.include_subdirectories_switch = ctk.CTkSwitch(self.frame, text="", variable=self.include_subdirectories_var)
        self.include_subdirectories_switch.grid(row=8, column=1, sticky="w", padx=5, pady=5)

    def _create_preview_option(self):
        """Create a toggle for test mode (only process current image)"""
        self.preview_mode_label = ctk.CTkLabel(self.frame, text="Test Run", width=100)
        self.preview_mode_label.grid(row=9, column=0, sticky="w", padx=5, pady=5)
        self.preview_mode_switch = ctk.CTkSwitch(self.frame, text="", variable=self.preview_mode_var)
        self.preview_mode_switch.grid(row=9, column=1, sticky="w", padx=5, pady=5)
        ToolTip(self.preview_mode_switch, "Masks only the current image to more quickly get feedback")

    def _create_progress_indicators(self):
        self.progress_label = ctk.CTkLabel(self.frame, text="Progress: 0/0", width=100)
        self.progress_label.grid(row=10, column=0, sticky="w", padx=5, pady=5)
        self.progress = ctk.CTkProgressBar(self.frame, orientation="horizontal", mode="determinate", width=200)
        self.progress.grid(row=10, column=1, sticky="w", padx=5, pady=5)

    def _create_action_buttons(self):
        self.create_masks_button = ctk.CTkButton(self.frame, text="Create Masks", width=310, command=self.create_masks)
        self.create_masks_button.grid(row=11, column=0, columnspan=2, sticky="w", padx=5, pady=5)

    def validate_inputs(self) -> tuple[bool, str]:
        """Validate all user inputs. Returns (is_valid, error_message)."""
        try:
            if not 0 <= (_threshold := float(self.threshold_entry.get())) <= 0.90:
                return False, "Threshold must be between 0.0 and 0.9 for usable results"
            if not 0 <= (_smooth_pixels := int(self.smooth_entry.get())) <= 10:
                return False, "Smooth pixels should be between 0 and 10"
            if not 0 <= (_expand_pixels := int(self.expand_entry.get())) <= 64:
                return False, "Expand pixels should be between 0 and 64"
            if not 0 <= (_alpha := float(self.alpha_entry.get())) <= 1:
                return False, "Alpha must be between 0.0 and 1.0"
            if not (_prompt := self.prompt_entry.get().strip()):
                return False, "Please enter a detection prompt"
            if not (_path := Path(self.path_entry.get())).is_dir():
                return False, "Please select a valid folder"
            return True, ""
        except ValueError as e:
            logger.exception("Validation error")
            return False, f"Invalid input: {e}"

    def browse_for_path(self, entry_box):
        path = filedialog.askdirectory()
        entry_box.focus_set()
        entry_box.delete(0, END)
        entry_box.insert(0, path)
        self.focus_set()

    def set_progress(self, value, max_value):
        progress = value / max_value
        self.progress.set(progress)
        self.progress_label.configure(text=f"{value}/{max_value}")
        self.progress.update()

    def create_masks(self):
        is_valid, error = self.validate_inputs()
        if not is_valid:
            messagebox.showwarning("Invalid Input", error)
            return

        self.create_masks_button.configure(state="disabled", text="Processing...")
        try:
            # Submit the task to the executor
            future = self._executor.submit(self._run_masking_process,
                self.prompt_entry.get(),
                float(self.alpha_entry.get()),
                float(self.threshold_entry.get()),
                int(self.smooth_entry.get()),
                int(self.expand_entry.get())
            )
            # Schedule a callback to handle completion/errors
            self.after(100, lambda: self._check_future(future))
        except Exception as e:
            logger.exception("Failed to start masking thread")
            self.create_masks_button.configure(state="normal", text="Create Masks")
            messagebox.showerror("Thread Error", str(e))

    def _check_future(self, future):
        if future.done():
            exc = future.exception()
            if exc:
                logger.exception("Error in masking process", exc_info=exc)
                self.create_masks_button.configure(state="normal", text="Create Masks")
                messagebox.showerror("Processing Error", f"An error occurred during mask creation:\n{exc}")
            # else: success is handled by _run_masking_process/_update_ui_after_processing
        else:
            # Not done yet, check again after 100ms
            self.after(100, lambda: self._check_future(future))

    def get_mode(self, mode_str: str) -> str:
        mode_map = {
            "Replace all masks": "replace",
            "Create if absent": "fill",
            "Add to existing": "add",
            "Subtract from existing": "subtract",
            "Blend with existing": "blend",
        }
        try:
            return mode_map[mode_str]
        except ValueError as e:
            logger.exception("Validation error")
            raise RuntimeError(f"Invalid input: {e}") from e

    def _run_masking_process(self, prompt, alpha, threshold, smooth_pixels, expand_pixels):
        try:
            masking_model = self.parent.model_manager.load_masking_model(
                self.model_var.get()
            )

            # Skip processing if model failed to load
            if masking_model is None:
                print(f"Failed to load masking model: {self.model_var.get()}")
                return

            # Set model_manager reference if this is a SAMdreamMaskModel
            if hasattr(masking_model, "__class__") and hasattr(masking_model.__class__, "__name__"):
                if masking_model.__class__.__name__ == "SAMdreamMaskModel":
                    masking_model.model_manager = self.parent.model_manager

            mode = self.get_mode(self.mode_var.get())

            # Check if preview mode is enabled
            if self.preview_mode_var.get() and hasattr(self.parent, "current_image_index"):
                # Get the current image path from parent window
                if (hasattr(self.parent, "image_rel_paths") and
                    0 <= self.parent.current_image_index < len(self.parent.image_rel_paths)):

                    # Get the full path to the current image
                    current_image_rel_path = self.parent.image_rel_paths[self.parent.current_image_index]
                    logger.info(f"Preview mode: Processing only image {current_image_rel_path}")

                    # Process just this single image
                    # We'll pass the sample_dir as the parent directory, but also pass the full relative path
                    # for precise filtering
                    masking_model.mask_folder(
                        sample_dir=self.parent.dir,
                        prompts=[prompt],
                        mode=mode,
                        alpha=alpha,
                        threshold=threshold,
                        smooth_pixels=smooth_pixels,
                        expand_pixels=expand_pixels,
                        progress_callback=self.set_progress,
                        include_subdirectories=True,  # Ensure we can find the file in subdirectories
                        single_file=current_image_rel_path  # Pass the relative path, not just the filename
                    )
                else:
                    messagebox.showwarning("Preview Error", "No current image is selected")
                    self.after(100, lambda: self.create_masks_button.configure(state="normal", text="Create Masks"))
                    return
            else:
                # Normal folder processing
                masking_model.mask_folder(
                    sample_dir=self.path_entry.get(),
                    prompts=[prompt],
                    mode=mode,
                    alpha=alpha,
                    threshold=threshold,
                    smooth_pixels=smooth_pixels,
                    expand_pixels=expand_pixels,
                    progress_callback=self.set_progress,
                    include_subdirectories=self.include_subdirectories_var.get(),
                )

            # Use after to safely update UI from the main thread
            self.after(100, self._update_ui_after_processing)
        except Exception as e:
            error_message = str(e)
            print(f"Error during masking process: {error_message}")
            # Use after to safely update UI from the main thread
            self.after(100, lambda: self._handle_masking_error(error_message))

    def _update_ui_after_processing(self):
        # Re-enable the create button
        self.create_masks_button.configure(state="normal", text="Create Masks")

        # Reload the current image data using image_handler
        if hasattr(self.parent, "image_handler") and hasattr(
            self.parent.image_handler, "load_image_data"
        ):
            self.parent.image_handler.load_image_data()
            self.parent.refresh_ui()

    def _handle_masking_error(self, error_message):
        self.create_masks_button.configure(state="normal", text="Create Masks")
        messagebox.showerror("Processing Error", f"An error occurred during mask creation:\n{error_message}")
