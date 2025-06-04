import logging
import os
import threading
from tkinter import TclError, filedialog, messagebox
from typing import Any, TypedDict

from modules.module.captioning.captioning_util import (
    filter_blacklisted_tags,
    get_blacklist_tags,
)

import customtkinter as ctk

logger = logging.getLogger(__name__)

class GridOptions(TypedDict, total=False):
    """TypedDict for grid layout options."""
    row: int
    column: int
    sticky: str
    padx: tuple[int, int] | int
    pady: int | tuple[int, int]
    columnspan: int
    rowspan: int

class GenerateCaptionsWindow(ctk.CTkToplevel):
    # Standard UI configuration constants
    LABEL_GRID: GridOptions = {"sticky": "w", "padx": (10, 5), "pady": 5}
    WIDGET_GRID: GridOptions = {"sticky": "ew", "padx": (5, 10), "pady": 5}
    EXPLANATION_FONT: tuple = ("", 12)
    EXPLANATION_WRAPLENGTH: int = 330
    DEFAULT_PADX: tuple[int, int] = (10, 10)
    DEFAULT_PADY: int = 5

    def __init__(
        self,
        parent: Any,
        path: str,
        parent_include_subdirectories: bool,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(parent, *args, **kwargs)
        self.parent = parent
        self.config_state: dict[str, Any] = {}
        self._threshold_visible: bool = False
        self._moondream_options_visible: bool = False

        self._setup_window("Batch generate captions", "380x540")
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.config_state["models"] = self.parent.model_manager.get_available_captioning_models()
        self.config_state["model_var"] = ctk.StringVar(
            self, self.config_state["models"][0] if self.config_state["models"] else ""
        )
        self.config_state["modes"] = [
            "Replace all captions",
            "Create if absent",
            "Add as new line",
        ]
        self.config_state["mode_mapping"] = {
            "Replace all captions": "replace",
            "Create if absent": "fill",
            "Add as new line": "add",
        }
        self.config_state["mode_var"] = ctk.StringVar(self, "Create if absent")

        # Add caption length options for Moondream2
        self.config_state["caption_lengths"] = ["short", "normal"]
        self.config_state["caption_length_var"] = ctk.StringVar(self, "normal")

        self._create_layout(path, parent_include_subdirectories)

    def _setup_window(self, title: str, geometry: str) -> None:
        self.title(title)
        self.geometry(geometry)
        self.minsize(400, 400)
        self.resizable(True, True)
        self.wait_visibility()
        self.grab_set()
        self.focus_set()

    def _create_layout(self, path: str, include_subdirectories: bool) -> None:
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        self.basic_options_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.caption_options_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.threshold_frame = ctk.CTkFrame(self.main_frame)
        self.moondream_options_frame = ctk.CTkFrame(self.main_frame)
        self.additional_options_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.progress_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.buttons_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")

        # Configure frame layouts using standard options
        frames = [
            self.basic_options_frame,
            self.caption_options_frame,
            self.additional_options_frame,
            self.progress_frame,
            self.buttons_frame,
        ]

        for i, frame in enumerate(frames):
            frame.grid(row=i, column=0, columnspan=2, sticky="ew", pady=self.DEFAULT_PADY)
            self._configure_standard_frame(frame)

        # Configure special frames
        self._configure_standard_frame(self.threshold_frame)
        self._configure_standard_frame(self.moondream_options_frame)

        self._create_basic_options(path)
        self._create_caption_configuration()
        self._create_threshold_configuration()
        self._create_moondream_configuration()
        self._create_additional_options(include_subdirectories)
        self._create_progress_indicators()
        self._create_action_buttons()

        self.config_state["model_var"].trace_add("write", lambda *args: self._update_model_specific_options())
        self._update_model_specific_options()

    def _configure_standard_frame(self, frame: ctk.CTkFrame) -> None:
        """Configure a frame with standard column weights."""
        frame.grid_columnconfigure(0, weight=0, minsize=120)
        frame.grid_columnconfigure(1, weight=1)

    def _configure_frame(self, frame: ctk.CTkFrame) -> None:
        """Legacy method maintained for backward compatibility."""
        self._configure_standard_frame(frame)

    def _create_labeled_widget(
        self,
        parent: ctk.CTkFrame,
        label_text: str,
        widget: Any,
        row: int,
        widget_options: GridOptions | None = None,
        label_options: GridOptions | None = None,
    ) -> None:
        """Create a labeled widget with standard grid options."""
        # Start with default options
        label_grid = self.LABEL_GRID.copy()
        label_grid["row"] = row
        label_grid["column"] = 0

        # Override with custom options if provided
        if label_options:
            label_grid.update(label_options)

        # Create and place the label
        label = ctk.CTkLabel(parent, text=label_text, anchor="w")
        label.grid(**label_grid)

        # Start with default widget options
        widget_grid = self.WIDGET_GRID.copy()
        widget_grid["row"] = row
        widget_grid["column"] = 1

        # Override with custom options if provided
        if widget_options:
            widget_grid.update(widget_options)

        # Place the widget
        widget.grid(**widget_grid)

    def _create_explanation_label(
        self,
        parent: ctk.CTkFrame,
        text: str,
        row: int,
        columnspan: int = 2,
        wraplength: int | None = None,
    ) -> ctk.CTkLabel:
        """Create a standardized explanation label."""
        label = ctk.CTkLabel(
            parent,
            text=text,
            font=self.EXPLANATION_FONT,
            wraplength=wraplength or self.EXPLANATION_WRAPLENGTH,
            justify="left",
        )
        label.grid(
            row=row,
            column=0,
            columnspan=columnspan,
            sticky="w",
            padx=self.DEFAULT_PADX[0],
            pady=self.DEFAULT_PADY
        )
        return label

    def _create_basic_options(self, path: str) -> None:
        self.model_dropdown = ctk.CTkOptionMenu(
            self.basic_options_frame,
            variable=self.config_state["model_var"],
            values=self.config_state["models"],
            dynamic_resizing=False,
            width=220,
        )
        self._create_labeled_widget(self.basic_options_frame, "Model", self.model_dropdown, row=0)

        # Path selection with browse button
        path_frame = ctk.CTkFrame(self.basic_options_frame, fg_color="transparent")
        path_frame.grid(row=1, column=1, **self.WIDGET_GRID)
        path_frame.grid_columnconfigure(0, weight=1)

        self.path_entry = ctk.CTkEntry(path_frame)
        self.path_entry.insert(0, path)
        self.path_entry.grid(row=0, column=0, sticky="ew", padx=(0, 5))

        self.path_button = ctk.CTkButton(
            path_frame,
            width=30,
            text="...",
            command=lambda: self._browse(entry_box=self.path_entry, directory=True),
        )
        self.path_button.grid(row=0, column=1, sticky="e")

        # Path label
        path_label = ctk.CTkLabel(self.basic_options_frame, text="Folder", anchor="w")
        path_label.grid(row=1, column=0, **self.LABEL_GRID)

        # Mode dropdown
        self.mode_dropdown = ctk.CTkOptionMenu(
            self.basic_options_frame,
            variable=self.config_state["mode_var"],
            values=self.config_state["modes"],
            dynamic_resizing=False,
            width=220,
        )
        self._create_labeled_widget(self.basic_options_frame, "Mode", self.mode_dropdown, row=2)

    def _create_caption_configuration(self) -> None:
        self.caption_entry = ctk.CTkEntry(self.caption_options_frame)
        self._create_labeled_widget(self.caption_options_frame, "Initial Caption", self.caption_entry, row=0)

        self.prefix_entry = ctk.CTkEntry(self.caption_options_frame)
        self._create_labeled_widget(self.caption_options_frame, "Caption Prefix", self.prefix_entry, row=1)

        self.postfix_entry = ctk.CTkEntry(self.caption_options_frame)
        self._create_labeled_widget(self.caption_options_frame, "Caption Suffix", self.postfix_entry, row=2)

        # Blacklist with browse button
        blacklist_frame = ctk.CTkFrame(self.caption_options_frame, fg_color="transparent")
        blacklist_frame.grid(row=3, column=1, **self.WIDGET_GRID)
        blacklist_frame.grid_columnconfigure(0, weight=1)

        self.blacklist_entry = ctk.CTkEntry(blacklist_frame)
        self.blacklist_entry.grid(row=0, column=0, sticky="ew", padx=(0, 5))

        self.blacklist_button = ctk.CTkButton(
            blacklist_frame,
            width=30,
            text="...",
            command=lambda: self._browse(
                entry_box=self.blacklist_entry,
                filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv"), ("All files", "*.*")]
            ),
        )
        self.blacklist_button.grid(row=0, column=1, sticky="e")

        # Blacklist label
        blacklist_label = ctk.CTkLabel(self.caption_options_frame, text="Blacklist", anchor="w")
        blacklist_label.grid(row=3, column=0, **self.LABEL_GRID)

        # Regex checkbox
        self.regex_enabled_var = ctk.BooleanVar(self, False)
        self.regex_enabled_checkbox = ctk.CTkCheckBox(
            self.caption_options_frame,
            text="Enable regex matching for blacklist",
            variable=self.regex_enabled_var,
        )
        widget_grid_no_sticky = {k: v for k, v in self.WIDGET_GRID.items() if k != 'sticky'}
        self.regex_enabled_checkbox.grid(row=4, column=1, sticky="w", **widget_grid_no_sticky)

        # Help text
        help_text = "Enter tags to blacklist, separated by commas. You can also specify a .txt or .csv file path."
        self._create_explanation_label(self.caption_options_frame, help_text, row=5)

    def _create_threshold_configuration(self) -> None:
        self.threshold_frame.grid_columnconfigure(0, weight=0, minsize=120)
        self.threshold_frame.grid_columnconfigure(1, weight=1, minsize=60)
        self.threshold_frame.grid_columnconfigure(2, weight=0)

        # General threshold controls
        self.general_threshold_label = ctk.CTkLabel(self.threshold_frame, text="General Tag Threshold", anchor="w")
        self.general_threshold_label.grid(row=0, column=0, **self.LABEL_GRID)

        self.general_threshold_var = ctk.StringVar(self, "0.35")
        self.general_threshold_entry = ctk.CTkEntry(
            self.threshold_frame, width=70, textvariable=self.general_threshold_var
        )
        self.general_threshold_entry.grid(row=0, column=1, sticky="w", padx=5, pady=5)

        self.general_mcut_var = ctk.BooleanVar(self, False)
        self.general_mcut_checkbox = ctk.CTkCheckBox(
            self.threshold_frame,
            text="MCut",
            variable=self.general_mcut_var,
            command=self._update_threshold_states,
        )
        self.general_mcut_checkbox.grid(row=0, column=2, sticky="w", padx=5, pady=5)

        # Character threshold controls
        self.character_threshold_label = ctk.CTkLabel(
            self.threshold_frame, text="Character Tag Threshold", anchor="w"
        )
        self.character_threshold_label.grid(row=1, column=0, **self.LABEL_GRID)

        self.character_threshold_var = ctk.StringVar(self, "0.85")
        self.character_threshold_entry = ctk.CTkEntry(
            self.threshold_frame, width=70, textvariable=self.character_threshold_var
        )
        self.character_threshold_entry.grid(row=1, column=1, sticky="w", padx=5, pady=5)

        self.character_mcut_var = ctk.BooleanVar(self, False)
        self.character_mcut_checkbox = ctk.CTkCheckBox(
            self.threshold_frame,
            text="MCut",
            variable=self.character_mcut_var,
            command=self._update_threshold_states,
        )
        self.character_mcut_checkbox.grid(row=1, column=2, sticky="w", padx=5, pady=5)

        # Explanation text
        explanation = (
            "MCut automatically determines optimal thresholds by finding the largest gap between adjacent "
            "label relevance scores. Enabling it disables the threshold inputs."
        )
        self._create_explanation_label(self.threshold_frame, explanation, row=2, columnspan=3)

        self._update_threshold_states()

    def _create_moondream_configuration(self) -> None:
        """Create configuration options specific to Moondream2 model"""
        # Caption length dropdown
        self.caption_length_label = ctk.CTkLabel(
            self.moondream_options_frame, text="Caption Length", anchor="w"
        )
        self.caption_length_label.grid(row=0, column=0, **self.LABEL_GRID)

        self.caption_length_dropdown = ctk.CTkOptionMenu(
            self.moondream_options_frame,
            variable=self.config_state["caption_length_var"],
            values=self.config_state["caption_lengths"],
            dynamic_resizing=False,
            width=120,
        )
        self.caption_length_dropdown.grid(row=0, column=1, **self.WIDGET_GRID)

        # Explanation text
        explanation = "Choose 'short' for a brief description or 'normal' for a more detailed caption."
        self._create_explanation_label(self.moondream_options_frame, explanation, row=1)

    def _create_additional_options(self, include_subdirectories: bool) -> None:
        self.include_subdirectories_label = ctk.CTkLabel(
            self.additional_options_frame, text="Include subfolders", anchor="w"
        )
        self.include_subdirectories_label.grid(row=0, column=0, sticky="w", padx=(10, 5), pady=5)
        self.include_subdirectories_var = ctk.BooleanVar(self, include_subdirectories)
        self.include_subdirectories_switch = ctk.CTkSwitch(
            self.additional_options_frame, text="", variable=self.include_subdirectories_var
        )
        self.include_subdirectories_switch.grid(row=0, column=1, sticky="w", padx=(5, 10), pady=5)

    def _create_progress_indicators(self) -> None:
        self.progress_label = ctk.CTkLabel(self.progress_frame, text="Progress: 0/0", anchor="w")
        self.progress_label.grid(row=0, column=0, sticky="w", padx=(10, 5), pady=5)
        self.progress = ctk.CTkProgressBar(self.progress_frame, orientation="horizontal", mode="determinate")
        self.progress.grid(row=0, column=1, sticky="ew", padx=(5, 10), pady=5)
        self.progress.set(0)

    def _create_action_buttons(self) -> None:
        self.create_captions_button = ctk.CTkButton(
            self.buttons_frame, text="Create Captions", command=self.create_captions
        )
        self.create_captions_button.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=5)

    def _browse(
        self,
        entry_box: ctk.CTkEntry,
        *,
        directory: bool = False,
        filetypes: list[tuple[str, str]] | None = None
    ) -> None:
        if directory:
            selected = filedialog.askdirectory()
        else:
            selected = filedialog.askopenfilename(
                filetypes=filetypes or [("All files", "*.*")]
            )
        if selected:
            entry_box.delete(0, "end")
            entry_box.insert(0, selected)
        self.focus_set()

    def _is_wd_model(self, model_name: str) -> bool:
        """Check if the model is a WD (WD14/SwinV2/EVA02) model."""
        return any(name in model_name for name in ["WD SwinV2", "WD EVA02", "WD14 VIT"]) or "WD" in model_name

    def _is_moondream_model(self, model_name: str) -> bool:
        """Check if the model is a Moondream model."""
        return "MOONDREAM" in model_name.upper()

    def _is_eva02_model(self, model_name: str) -> bool:
        """Check if the model is specifically an EVA02 model."""
        return "EVA02" in model_name

    def _update_model_specific_options(self, *args: Any) -> None:
        """Update the visibility of model-specific option frames based on selected model"""
        model = self.config_state["model_var"].get()
        threshold_height = 150
        moondream_height = 110  # Height of the Moondream options frame

        # Determine which frames should be visible
        is_supported_wd_model = self._is_wd_model(model)
        is_moondream_model = self._is_moondream_model(model)

        # Get current window dimensions
        geometry = self.geometry().split("+")[0]
        current_width, current_height = map(int, geometry.split("x"))
        new_height = current_height

        # Handle Threshold frame visibility
        if is_supported_wd_model and not self._threshold_visible:
            new_height += threshold_height
            self._threshold_visible = True
            self.threshold_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=5)
            row_offset = 1
        elif not is_supported_wd_model and self._threshold_visible:
            new_height -= threshold_height
            self._threshold_visible = False
            self.threshold_frame.grid_remove()
            row_offset = 0
        else:
            row_offset = 1 if self._threshold_visible else 0

        # Handle Moondream options frame visibility
        if is_moondream_model and not self._moondream_options_visible:
            new_height += moondream_height
            self._moondream_options_visible = True
            self.moondream_options_frame.grid(row=2+row_offset, column=0, columnspan=2, sticky="ew", pady=5)
            row_offset += 1
        elif not is_moondream_model and self._moondream_options_visible:
            new_height -= moondream_height
            self._moondream_options_visible = False
            self.moondream_options_frame.grid_remove()
        else:
            row_offset += 1 if self._moondream_options_visible else 0

        # Update positions of remaining frames
        self.additional_options_frame.grid(row=2+row_offset, column=0, columnspan=2, sticky="ew", pady=5)
        self.progress_frame.grid(row=3+row_offset, column=0, columnspan=2, sticky="ew", pady=5)
        self.buttons_frame.grid(row=4+row_offset, column=0, columnspan=2, sticky="ew", pady=5)

        # Update window size
        self.geometry(f"{current_width}x{new_height}")

        # For backward compatibility, continue to update threshold states
        self._update_threshold_states()

    def _update_threshold_visibility(self, *args: Any) -> None:
        """Legacy method maintained for backward compatibility"""
        self._update_model_specific_options()

    def _update_threshold_states(self) -> None:
        if self.general_mcut_var.get():
            self.general_threshold_var.set("0.0")
            self.general_threshold_entry.configure(state="disabled", placeholder_text="Auto")
            self.general_threshold_label.configure(text_color=("gray", "gray"))
        else:
            model = self.config_state["model_var"].get()
            self.general_threshold_var.set("0.5" if self._is_eva02_model(model) else "0.35")
            self.general_threshold_entry.configure(state="normal", placeholder_text="")
            self.general_threshold_label.configure(text_color=("black", "white"))
        if self.character_mcut_var.get():
            self.character_threshold_var.set("0.15")
            self.character_threshold_entry.configure(state="disabled", placeholder_text="Auto")
            self.character_threshold_label.configure(text_color=("gray", "gray"))
        else:
            self.character_threshold_var.set("0.85")
            self.character_threshold_entry.configure(state="normal", placeholder_text="")
            self.character_threshold_label.configure(text_color=("black", "white"))

    def _window_exists(self) -> bool:
        """Check if the window still exists and is valid."""
        try:
            return self.winfo_exists() == 1
        except TclError:
            return False

    def _safely_update_widget(self, widget: Any, **kwargs: Any) -> None:
        """Safely update a widget if the window still exists."""
        try:
            if self._window_exists():
                widget.configure(**kwargs)
                widget.update()
        except (TclError, RuntimeError):
            # Pass silently if the widget is no longer valid
            pass

    def set_progress(self, value: int, max_value: int) -> None:
        """Update progress bar safely from any thread."""
        progress = value / max_value if max_value > 0 else 0

        # Use after() to schedule UI update on main thread
        try:
            if self._window_exists():
                self.after(0, lambda: self._update_progress_ui(progress, value, max_value))
        except (TclError, RuntimeError):
            pass

    def _update_progress_ui(self, progress: float, value: int, max_value: int) -> None:
        """Update progress UI elements on the main thread."""
        try:
            if self._window_exists():
                self.progress.set(progress)
                self.progress_label.configure(text=f"{value}/{max_value}")
                self.progress.update()
        except (TclError, RuntimeError):
            pass

    def create_captions(self) -> None:
        """Create captions with input validation."""
        # Validate model selection
        model_name = self.config_state["model_var"].get()
        if not model_name:
            self._show_error("No model selected", "Please select a captioning model.")
            return

        # Validate path exists
        path = self.path_entry.get().strip()
        if not path or not os.path.exists(path) or not os.path.isdir(path):
            self._show_error("Invalid path", "The specified folder does not exist.")
            return

        # Model-specific kwargs initialization
        model_kwargs: dict[str, Any] = {}

        # Set model-specific parameters based on model type with validation
        if self._is_wd_model(model_name):
            try:
                # Validate general threshold
                general_threshold_str = self.general_threshold_var.get()
                if not self.general_mcut_var.get() and not self._validate_float_range(general_threshold_str, 0, 1):
                    self._show_error(
                        "Invalid general threshold",
                        "General threshold must be a number between 0 and 1."
                    )
                    return
                min_general = 0.0 if self.general_mcut_var.get() else float(general_threshold_str)

                # Validate character threshold
                character_threshold_str = self.character_threshold_var.get()
                if not self.character_mcut_var.get() and not self._validate_float_range(character_threshold_str, 0, 1):
                    self._show_error(
                        "Invalid character threshold",
                        "Character threshold must be a number between 0 and 1."
                    )
                    return
                min_character = 0.15 if self.character_mcut_var.get() else float(character_threshold_str)

                model_kwargs = {
                    "use_mcut_general": self.general_mcut_var.get(),
                    "use_mcut_character": self.character_mcut_var.get(),
                    "min_general_threshold": min_general,
                    "min_character_threshold": min_character,
                }
            except ValueError as e:
                logger.warning(f"Invalid threshold value: {e}")
                self._show_error("Invalid threshold", f"Could not parse threshold value: {e}")
                return
        elif self._is_moondream_model(model_name):
            caption_length = self.config_state["caption_length_var"].get()
            # Validate caption length from MoonDream
            if caption_length not in self.config_state["caption_lengths"]:
                self._show_error(
                    "Invalid caption length",
                    f"Caption length must be one of: {', '.join(self.config_state['caption_lengths'])}"
                )
                return

            model_kwargs = {
                "caption_length": caption_length,
                "stream": True
            }

        # Load or initialize the appropriate model
        try:
            if self._is_wd_model(model_name):
                self.parent.model_manager.captioning_model = None
                self.parent.model_manager.current_captioning_model_name = None
                model_class = self.parent.model_manager._captioning_registry[model_name]
                captioning_model = model_class(
                    self.parent.model_manager.device,
                    self.parent.model_manager.precision,
                    model_name=model_name,
                    **model_kwargs,
                )
                self.parent.model_manager.captioning_model = captioning_model
                self.parent.model_manager.current_captioning_model_name = model_name
            else:
                # For other models including Moondream2
                captioning_model = self.parent.model_manager.load_captioning_model(model_name, **model_kwargs)
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            self._show_error("Model loading error", f"Failed to load model {model_name}: {str(e)}")
            return

        if captioning_model is None:
            logger.error(f"Failed to load model: {model_name}")
            self._show_error("Model error", f"Failed to load model: {model_name}")
            return

        # Get the blacklist once at the beginning to avoid caching issues
        blacklist_text = self.blacklist_entry.get().strip()
        blacklist_tags = []
        if blacklist_text:
            try:
                blacklist_tags = get_blacklist_tags(blacklist_text, model_name)
                logger.info(f"Loaded blacklist tags: {blacklist_tags}")
            except Exception as e:
                logger.error(f"Failed to load blacklist tags: {e}")
                self._show_error("Blacklist error", f"Failed to load blacklist tags: {str(e)}")
                return

        # Store the original generate_caption method
        original_generate_caption = captioning_model.generate_caption

        def generate_caption_with_blacklist(caption_sample, initial, initial_caption, caption_prefix, caption_postfix):
            caption = original_generate_caption(caption_sample, initial, initial_caption, caption_prefix, caption_postfix)
            filtered_caption = filter_blacklisted_tags(
                caption,
                blacklist_text,
                model_name,
                self.regex_enabled_var.get()
            )
            logger.debug(f"Original caption: {caption}")
            logger.debug(f"Filtered caption: {filtered_caption}")
            return filtered_caption

        # Replace generate_caption with our wrapper
        captioning_model.generate_caption = generate_caption_with_blacklist

        # Disable the button while processing
        self._safely_update_widget(self.create_captions_button, state="disabled", text="Processing...")

        # Get all parameters before starting the thread
        mode = self.config_state["mode_mapping"][self.config_state["mode_var"].get()]
        initial_caption = self.caption_entry.get()
        caption_prefix = self.prefix_entry.get()
        caption_postfix = self.postfix_entry.get()
        include_subdirectories = self.include_subdirectories_var.get()

        # Create a thread for the captioning process
        def caption_thread_func():
            try:
                captioning_model.caption_folder(
                    sample_dir=path,
                    initial_caption=initial_caption,
                    caption_prefix=caption_prefix,
                    caption_postfix=caption_postfix,
                    mode=mode,
                    progress_callback=self.set_progress,
                    include_subdirectories=include_subdirectories,
                )
                # Schedule success message on the main thread
                if self._window_exists():
                    self.after(0, lambda: self._show_info("Process complete", "Caption generation completed successfully."))

                # Schedule UI refresh on the main thread
                if self._window_exists():
                    self.after(0, self._refresh_parent_ui)

            except Exception as exception:
                logger.error(f"Error during caption generation: {exception}")
                # Capture the error message before using it in the lambda
                error_message = str(exception)
                # Schedule error message on the main thread
                if self._window_exists():
                    self.after(0, lambda: self._show_error("Caption generation error", f"Error during processing: {error_message}"))
            finally:
                # Restore the original generate_caption method
                captioning_model.generate_caption = original_generate_caption
                # Re-enable the button on the main thread
                if self._window_exists():
                    self.after(0, lambda: self._safely_update_widget(
                        self.create_captions_button, state="normal", text="Create Captions"))

        # Start the captioning thread
        caption_thread = threading.Thread(target=caption_thread_func, daemon=True)
        caption_thread.start()

    def _refresh_parent_ui(self) -> None:
        """Refresh parent UI safely."""
        try:
            if self._window_exists() and hasattr(self.parent, "image_handler") and hasattr(self.parent.image_handler, "load_image_data"):
                self.parent.image_handler.load_image_data()
                self.parent.refresh_ui()
        except (TclError, RuntimeError, AttributeError):
            # Silently pass if window is destroyed or methods don't exist
            pass

    def _validate_float_range(self, value: str, min_val: float, max_val: float) -> bool:
        """Validate if a string represents a float within the specified range."""
        try:
            float_val = float(value)
            return min_val <= float_val <= max_val
        except ValueError:
            return False

    def _show_error(self, title: str, message: str) -> None:
        """Display an error dialog using CTk."""
        messagebox.showerror(title, message)
        try:
            if self._window_exists():
                self.focus_set()
        except (TclError, RuntimeError):
            pass

    def _show_info(self, title: str, message: str) -> None:
        """Display an information dialog using CTk."""
        messagebox.showinfo(title, message)
        try:
            if self._window_exists():
                self.focus_set()
        except (TclError, RuntimeError):
            pass
