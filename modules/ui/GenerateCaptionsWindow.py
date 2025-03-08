import csv
import functools
import os
import re
from tkinter import filedialog

import customtkinter as ctk


@functools.lru_cache
def get_blacklist_tags(blacklist_text, model_name):
    """Convert blacklist_text to list depending on whether it's a file or a comma-separated string"""
    # Determine delimiter based on model type
    delimiter = "," if "WD" in model_name else None

    if blacklist_text.endswith(".txt") and os.path.isfile(blacklist_text):
        with open(blacklist_text) as blacklist_file:
            return [line.rstrip("\n") for line in blacklist_file]
    elif blacklist_text.endswith(".csv") and os.path.isfile(blacklist_text):
        with open(blacklist_text, "r") as blacklist_file:
            return [row[0] for row in csv.reader(blacklist_file)]
    elif delimiter:
        return [tag.strip() for tag in blacklist_text.split(delimiter)]
    else:
        # For non-WD models with no delimiter specified, just return the text as is
        return [blacklist_text]

def parse_regex_blacklist(blacklist_tags, caption_tags):
    """Match regex patterns in blacklist against caption tags"""
    matched_tags = []
    regex_spchars = set(".^$*+?!{}[]|()\\")

    for pattern in blacklist_tags:
        if any((char in regex_spchars) for char in pattern):
            # It's a regex pattern
            try:
                r = re.compile(pattern)
                for tag in caption_tags:
                    if r.fullmatch(tag) and tag not in matched_tags:
                        matched_tags.append(tag)
            except re.error:
                # Skip invalid regex patterns
                pass
        else:
            # It's a direct match
            if pattern in caption_tags and pattern not in matched_tags:
                matched_tags.append(pattern)

    return matched_tags


class GenerateCaptionsWindow(ctk.CTkToplevel):
    def __init__(
        self, parent, path, parent_include_subdirectories, *args, **kwargs
    ):
        ctk.CTkToplevel.__init__(self, parent, *args, **kwargs)
        self.parent = parent

        # Setup window properties
        self._setup_window("Batch generate captions", "400x540")

        # Configure main grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Get available models dynamically
        self.models = (
            self.parent.model_manager.get_available_captioning_models()
        )
        self.model_var = ctk.StringVar(
            self, self.models[0] if self.models else ""
        )

        # Define modes with mapping to API values
        self.modes = [
            "Replace all captions",
            "Create if absent",
            "Add as new line",
        ]
        self.mode_mapping = {
            "Replace all captions": "replace", #ovewrite
            "Create if absent": "fill",
            "Add as new line": "add", #append
        }
        self.mode_var = ctk.StringVar(self, "Create if absent")

        # Set up the UI
        self._create_layout(path, parent_include_subdirectories)

    def _setup_window(self, title, geometry):
        self.title(title)
        self.geometry(geometry)
        self.minsize(400, 400)  # Set minimum size
        self.resizable(True, True)
        self.wait_visibility()
        self.grab_set()
        self.focus_set()

    def _create_layout(self, path, parent_include_subdirectories):
        # Create main frame
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(
            row=0, column=0, sticky="nsew", padx=10, pady=10
        )

        # Configure main frame grid
        self.main_frame.grid_columnconfigure(0, weight=0)  # Label column
        self.main_frame.grid_columnconfigure(1, weight=1)  # Content column

        # Create separate frames for each section
        self.basic_options_frame = ctk.CTkFrame(
            self.main_frame, fg_color="transparent"
        )
        self.basic_options_frame.grid(
            row=0, column=0, columnspan=2, sticky="ew", padx=0, pady=5
        )

        self.caption_options_frame = ctk.CTkFrame(
            self.main_frame, fg_color="transparent"
        )
        self.caption_options_frame.grid(
            row=1, column=0, columnspan=2, sticky="ew", padx=0, pady=5
        )

        # Create the threshold frame but don't add to grid yet
        self.threshold_frame = ctk.CTkFrame(self.main_frame)

        # Set threshold visible flag to False initially
        self._threshold_visible = False

        self.additional_options_frame = ctk.CTkFrame(
            self.main_frame, fg_color="transparent"
        )
        self.additional_options_frame.grid(
            row=2, column=0, columnspan=2, sticky="ew", padx=0, pady=5
        )

        self.progress_frame = ctk.CTkFrame(
            self.main_frame, fg_color="transparent"
        )
        self.progress_frame.grid(
            row=3, column=0, columnspan=2, sticky="ew", padx=0, pady=5
        )

        self.buttons_frame = ctk.CTkFrame(
            self.main_frame, fg_color="transparent"
        )
        self.buttons_frame.grid(
            row=4, column=0, columnspan=2, sticky="ew", padx=0, pady=5
        )

        # Configure the frames' grid columns
        for frame in [
            self.basic_options_frame,
            self.caption_options_frame,
            self.threshold_frame,
            self.additional_options_frame,
            self.progress_frame,
            self.buttons_frame,
        ]:
            frame.grid_columnconfigure(
                0, weight=0, minsize=120
            )  # Fixed width for labels
            frame.grid_columnconfigure(1, weight=1)  # Content expands

        # Create UI components in each frame
        self._create_basic_options(path)
        self._create_caption_configuration()
        self._create_threshold_configuration()
        self._create_additional_options(parent_include_subdirectories)
        self._create_progress_indicators()
        self._create_action_buttons()

        # Initialize the visibility of threshold controls based on model
        self.model_var.trace("w", self._update_threshold_visibility)
        self._update_threshold_visibility()

    def _create_basic_options(self, path):
        # Model selection
        self.model_label = ctk.CTkLabel(
            self.basic_options_frame, text="Model", anchor="w"
        )
        self.model_label.grid(
            row=0, column=0, sticky="w", padx=(10, 5), pady=5
        )

        self.model_dropdown = ctk.CTkOptionMenu(
            self.basic_options_frame,
            variable=self.model_var,
            values=self.models,
            dynamic_resizing=False,
            width=220,
        )
        self.model_dropdown.grid(
            row=0, column=1, sticky="ew", padx=(5, 10), pady=5
        )

        # Path selection
        self.path_label = ctk.CTkLabel(
            self.basic_options_frame, text="Folder", anchor="w"
        )
        self.path_label.grid(
            row=1, column=0, sticky="w", padx=(10, 5), pady=5
        )

        path_frame = ctk.CTkFrame(
            self.basic_options_frame, fg_color="transparent"
        )
        path_frame.grid(row=1, column=1, sticky="ew", padx=(5, 10), pady=5)
        path_frame.grid_columnconfigure(0, weight=1)
        path_frame.grid_columnconfigure(1, weight=0)

        self.path_entry = ctk.CTkEntry(path_frame)
        self.path_entry.insert(0, path)
        self.path_entry.grid(row=0, column=0, sticky="ew", padx=(0, 5))

        self.path_button = ctk.CTkButton(
            path_frame,
            width=30,
            text="...",
            command=lambda: self.browse_for_path(self.path_entry),
        )
        self.path_button.grid(row=0, column=1, sticky="e")

        # Mode selection
        self.mode_label = ctk.CTkLabel(
            self.basic_options_frame, text="Mode", anchor="w"
        )
        self.mode_label.grid(
            row=2, column=0, sticky="w", padx=(10, 5), pady=5
        )

        self.mode_dropdown = ctk.CTkOptionMenu(
            self.basic_options_frame,
            variable=self.mode_var,
            values=self.modes,
            dynamic_resizing=False,
            width=220,
        )
        self.mode_dropdown.grid(
            row=2, column=1, sticky="ew", padx=(5, 10), pady=5
        )

    def _create_caption_configuration(self):
        # Initial caption
        self.caption_label = ctk.CTkLabel(
            self.caption_options_frame, text="Initial Caption", anchor="w"
        )
        self.caption_label.grid(
            row=0, column=0, sticky="w", padx=(10, 5), pady=5
        )

        self.caption_entry = ctk.CTkEntry(self.caption_options_frame)
        self.caption_entry.grid(
            row=0, column=1, sticky="ew", padx=(5, 10), pady=5
        )

        # Prefix
        self.prefix_label = ctk.CTkLabel(
            self.caption_options_frame, text="Caption Prefix", anchor="w"
        )
        self.prefix_label.grid(
            row=1, column=0, sticky="w", padx=(10, 5), pady=5
        )

        self.prefix_entry = ctk.CTkEntry(self.caption_options_frame)
        self.prefix_entry.grid(
            row=1, column=1, sticky="ew", padx=(5, 10), pady=5
        )

        # Postfix
        self.postfix_label = ctk.CTkLabel(
            self.caption_options_frame, text="Caption Postfix", anchor="w"
        )
        self.postfix_label.grid(
            row=2, column=0, sticky="w", padx=(10, 5), pady=5
        )

        self.postfix_entry = ctk.CTkEntry(self.caption_options_frame)
        self.postfix_entry.grid(
            row=2, column=1, sticky="ew", padx=(5, 10), pady=5
        )

        # Blacklist section
        self.blacklist_label = ctk.CTkLabel(
            self.caption_options_frame, text="Blacklist", anchor="w"
        )
        self.blacklist_label.grid(
            row=3, column=0, sticky="w", padx=(10, 5), pady=5
        )

        blacklist_frame = ctk.CTkFrame(
            self.caption_options_frame, fg_color="transparent"
        )
        blacklist_frame.grid(row=3, column=1, sticky="ew", padx=(5, 10), pady=5)
        blacklist_frame.grid_columnconfigure(0, weight=1)
        blacklist_frame.grid_columnconfigure(1, weight=0)

        self.blacklist_entry = ctk.CTkEntry(blacklist_frame)
        self.blacklist_entry.grid(row=0, column=0, sticky="ew", padx=(0, 5))

        self.blacklist_button = ctk.CTkButton(
            blacklist_frame,
            width=30,
            text="...",
            command=lambda: self.browse_for_blacklist(self.blacklist_entry),
        )
        self.blacklist_button.grid(row=0, column=1, sticky="e")

        # Regex toggle
        self.regex_enabled_var = ctk.BooleanVar(self, False)
        self.regex_enabled_checkbox = ctk.CTkCheckBox(
            self.caption_options_frame,
            text="Enable regex matching for blacklist",
            variable=self.regex_enabled_var,
        )
        self.regex_enabled_checkbox.grid(
            row=4, column=1, sticky="w", padx=(5, 10), pady=5
        )

        # Help text for blacklist
        blacklist_help = "Enter tags to blacklist, separated by commas. You can also specify a .txt or .csv file path."
        self.blacklist_help_label = ctk.CTkLabel(
            self.caption_options_frame,
            text=blacklist_help,
            font=("", 12),
            wraplength=350,
            justify="left",
        )
        self.blacklist_help_label.grid(
            row=5, column=0, columnspan=2, sticky="w", padx=10, pady=5
        )

    def _create_threshold_configuration(self):
        """Create threshold configuration options for WD models"""
        # Configure threshold frame columns
        self.threshold_frame.grid_columnconfigure(0, weight=0, minsize=120)
        self.threshold_frame.grid_columnconfigure(1, weight=1, minsize=60)
        self.threshold_frame.grid_columnconfigure(2, weight=0)

        # General tag threshold controls
        self.general_threshold_label = ctk.CTkLabel(
            self.threshold_frame, text="General Tag Threshold", anchor="w"
        )
        self.general_threshold_label.grid(
            row=0, column=0, sticky="w", padx=(10, 5), pady=5
        )

        self.general_threshold_var = ctk.StringVar(self, "0.35")
        self.general_threshold_entry = ctk.CTkEntry(
            self.threshold_frame,
            width=70,
            textvariable=self.general_threshold_var,
        )
        self.general_threshold_entry.grid(
            row=0, column=1, sticky="w", padx=5, pady=5
        )

        # MCut for general tags
        self.general_mcut_var = ctk.BooleanVar(self, False)
        self.general_mcut_checkbox = ctk.CTkCheckBox(
            self.threshold_frame,
            text="MCut",
            variable=self.general_mcut_var,
            command=self._update_threshold_states,
        )
        self.general_mcut_checkbox.grid(
            row=0, column=2, sticky="w", padx=5, pady=5
        )

        # Character tag threshold controls
        self.character_threshold_label = ctk.CTkLabel(
            self.threshold_frame,
            text="Character Tag Threshold",
            anchor="w",
        )
        self.character_threshold_label.grid(
            row=1, column=0, sticky="w", padx=(10, 5), pady=5
        )

        self.character_threshold_var = ctk.StringVar(self, "0.85")
        self.character_threshold_entry = ctk.CTkEntry(
            self.threshold_frame,
            width=70,
            textvariable=self.character_threshold_var,
        )
        self.character_threshold_entry.grid(
            row=1, column=1, sticky="w", padx=5, pady=5
        )

        # MCut for character tags
        self.character_mcut_var = ctk.BooleanVar(self, False)
        self.character_mcut_checkbox = ctk.CTkCheckBox(
            self.threshold_frame,
            text="MCut",
            variable=self.character_mcut_var,
            command=self._update_threshold_states,
        )
        self.character_mcut_checkbox.grid(
            row=1, column=2, sticky="w", padx=5, pady=5
        )

        # MCut explanation label
        mcut_explanation = "MCut automatically determines optimal thresholds by finding the largest gap between adjacent label relevance scores. Enabling it disables the threshold inputs."
        self.mcut_explanation_label = ctk.CTkLabel(
            self.threshold_frame,
            text=mcut_explanation,
            font=("", 12),
            wraplength=350,
            justify="left",
        )
        self.mcut_explanation_label.grid(
            row=2, column=0, columnspan=3, sticky="w", padx=10, pady=10
        )

        # Initialize threshold entry states
        self._update_threshold_states()

    def _create_additional_options(self, parent_include_subdirectories):
        self.include_subdirectories_label = ctk.CTkLabel(
            self.additional_options_frame,
            text="Include subfolders",
            anchor="w",
        )
        self.include_subdirectories_label.grid(
            row=0, column=0, sticky="w", padx=(10, 5), pady=5
        )

        self.include_subdirectories_var = ctk.BooleanVar(
            self, parent_include_subdirectories
        )
        self.include_subdirectories_switch = ctk.CTkSwitch(
            self.additional_options_frame,
            text="",
            variable=self.include_subdirectories_var,
        )
        self.include_subdirectories_switch.grid(
            row=0, column=1, sticky="w", padx=(5, 10), pady=5
        )

    def _create_progress_indicators(self):
        self.progress_label = ctk.CTkLabel(
            self.progress_frame, text="Progress: 0/0", anchor="w"
        )
        self.progress_label.grid(
            row=0, column=0, sticky="w", padx=(10, 5), pady=5
        )

        self.progress = ctk.CTkProgressBar(
            self.progress_frame,
            orientation="horizontal",
            mode="determinate",
        )
        self.progress.grid(
            row=0, column=1, sticky="ew", padx=(5, 10), pady=5
        )
        self.progress.set(0)

    def _create_action_buttons(self):
        self.create_captions_button = ctk.CTkButton(
            self.buttons_frame,
            text="Create Captions",
            command=self.create_captions,
        )
        self.create_captions_button.grid(
            row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=5
        )

    def browse_for_blacklist(self, entry_box):
        # get the path from the user
        path = filedialog.askopenfilename(
            filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if path:  # Only update if a path was selected
            # delete entry box text
            entry_box.focus_set()
            entry_box.delete(0, filedialog.END)
            entry_box.insert(0, path)
        self.focus_set()

    def _update_threshold_visibility(self, *args):
        """Show/hide threshold options based on selected model"""
        model = self.model_var.get()

        threshold_height = 150

        # Check if it's one of the specific Booru style models that benefit from MCut
        is_supported_wd_model = any(
            name in model for name in ["WD SwinV2", "WD EVA02", "WD14 VIT"]
        )

        if (
            not self.general_mcut_var.get()
        ):  # Only update if MCut is not enabled
            if "EVA02" in model:
                self.general_threshold_var.set("0.5")
            elif is_supported_wd_model:
                self.general_threshold_var.set("0.35")

        # Get current geometry as x, y, width, height
        geometry = self.geometry().split("+")
        size_parts = geometry[0].split("x")
        current_width = int(size_parts[0])
        current_height = int(size_parts[1])

        if is_supported_wd_model and not self._threshold_visible:
            # Calculate new height only - width remains the same
            new_height = current_height + threshold_height

            # Show threshold frame - insert it at the right position
            self.threshold_frame.grid(
                row=2, column=0, columnspan=2, sticky="ew", padx=0, pady=5
            )

            # Move other frames down
            self.additional_options_frame.grid(row=3)
            self.progress_frame.grid(row=4)
            self.buttons_frame.grid(row=5)

            # Force window to resize by explicitly setting new geometry
            # Keep the width exactly the same
            self.geometry(f"{current_width}x{new_height}")

            self._threshold_visible = True

        elif not is_supported_wd_model and self._threshold_visible:
            # Calculate new height only - width remains the same
            new_height = current_height - threshold_height

            # Hide threshold frame
            self.threshold_frame.grid_remove()

            # Move other frames up
            self.additional_options_frame.grid(row=2)
            self.progress_frame.grid(row=3)
            self.buttons_frame.grid(row=4)

            # Force window to resize by explicitly setting new geometry
            # Keep the width exactly the same
            self.geometry(f"{current_width}x{new_height}")

            self._threshold_visible = False

    def _update_threshold_states(self):
        """Update the state of threshold entries based on MCut checkbox states"""
        # General threshold entry
        if self.general_mcut_var.get():
            # Set the general threshold value to 0.0 when MCut is enabled
            self.general_threshold_var.set("0.0")
            self.general_threshold_entry.configure(
                state="disabled", placeholder_text="Auto"
            )
            self.general_threshold_label.configure(
                text_color=("gray", "gray")
            )
        else:
            # Reset general threshold to model-appropriate default value when MCut is disabled
            model = self.model_var.get()
            if "EVA02" in model:
                self.general_threshold_var.set("0.5")
            else:
                self.general_threshold_var.set("0.35")

            self.general_threshold_entry.configure(
                state="normal", placeholder_text=""
            )
            self.general_threshold_label.configure(
                text_color=("black", "white")  # dark mode, light mode
            )

        # Character threshold entry
        if self.character_mcut_var.get():
            # Set the character threshold value to 0.15 when MCut is enabled
            self.character_threshold_var.set("0.15")
            self.character_threshold_entry.configure(
                state="disabled", placeholder_text="Auto"
            )
            self.character_threshold_label.configure(
                text_color=("gray", "gray")
            )
        else:
            # Reset character threshold to default value when MCut is disabled
            self.character_threshold_var.set("0.85")
            self.character_threshold_entry.configure(
                state="normal", placeholder_text=""
            )
            self.character_threshold_label.configure(
                text_color=("black", "white")  # dark mode, light mode
            )

    def browse_for_path(self, entry_box):
        # get the path from the user
        path = filedialog.askdirectory()
        if path:  # Only update if a path was selected
            # delete entry box text
            entry_box.focus_set()
            entry_box.delete(0, filedialog.END)
            entry_box.insert(0, path)
        self.focus_set()

    def set_progress(self, value, max_value):
        progress = value / max_value if max_value > 0 else 0
        self.progress.set(progress)
        self.progress_label.configure(text=f"{value}/{max_value}")
        self.progress.update()

    def filter_blacklisted_tags(self, caption, model_name):
        """Remove blacklisted tags from caption"""
        blacklist_text = self.blacklist_entry.get().strip()
        if not blacklist_text:
            return caption  # No blacklist specified

        # Determine delimiter based on model type
        delimiter = "," if "WD" in model_name else " "

        # Get blacklist tags using standalone function
        blacklist_tags = get_blacklist_tags(blacklist_text, model_name)

        # Split caption into tags
        caption_tags = [tag.strip() for tag in caption.split(delimiter)]

        # Find tags to remove using standalone function
        if self.regex_enabled_var.get():
            tags_to_remove = parse_regex_blacklist(blacklist_tags, caption_tags)
        else:
            # Direct matching without regex
            tags_to_remove = [
                tag for tag in caption_tags if tag in blacklist_tags
            ]

        # Remove blacklisted tags
        filtered_tags = [
            tag for tag in caption_tags if tag not in tags_to_remove
        ]

        # Rejoin tags with appropriate delimiter
        return delimiter.join(filtered_tags)

    def create_captions(self):
        model_name = self.model_var.get()

      # Get thresholds and MCut settings if it's a WD model
        wd_model_kwargs = {}
        if "WD" in model_name:
            try:
                # Use ternary operator for cleaner code
                min_general = 0.0 if self.general_mcut_var.get() else float(self.general_threshold_var.get())

                wd_model_kwargs = {
                    "use_mcut_general": self.general_mcut_var.get(),
                    "use_mcut_character": self.character_mcut_var.get(),
                    "min_general_threshold": min_general,
                    "min_character_threshold": 0.15,  # Always keep minimum for character tags
                }
            except ValueError:
                print("Invalid threshold value, using defaults")
                wd_model_kwargs = {
                    "use_mcut_general": self.general_mcut_var.get(),
                    "use_mcut_character": self.character_mcut_var.get(),
                }

        # Load the model with appropriate parameters
        if "WD" in model_name:
            # First remove the current model if it's loaded, to ensure our parameters take effect
            self.parent.model_manager.captioning_model = None
            self.parent.model_manager.current_captioning_model_name = None

            # Create a new model with our parameters
            model_class = self.parent.model_manager._captioning_registry[
                model_name
            ]
            captioning_model = model_class(
                self.parent.model_manager.device,
                self.parent.model_manager.precision,
                model_name=model_name,
                **wd_model_kwargs,
            )

            # Store it in the model manager
            self.parent.model_manager.captioning_model = captioning_model
            self.parent.model_manager.current_captioning_model_name = (
                model_name
            )
        else:
            # Use standard model loading for non-WD models
            captioning_model = (
                self.parent.model_manager.load_captioning_model(model_name)
            )

        # Skip processing if model failed to load
        if captioning_model is None:
            print(f"Failed to load model: {model_name}")
            return

        # Save original caption_image method to restore later
        original_caption_image = captioning_model.caption_image

        # Create a wrapper function that applies blacklist filtering
        def caption_image_with_blacklist(*args, **kwargs):
            caption = original_caption_image(*args, **kwargs)
            return self.filter_blacklisted_tags(caption, model_name)

        # Replace the caption_image method with our filtered version
        captioning_model.caption_image = caption_image_with_blacklist

        mode = self.mode_mapping[self.mode_var.get()]

        try:
            captioning_model.caption_folder(
                sample_dir=self.path_entry.get(),
                initial_caption=self.caption_entry.get(),
                caption_prefix=self.prefix_entry.get(),
                caption_postfix=self.postfix_entry.get(),
                mode=mode,
                progress_callback=self.set_progress,
                include_subdirectories=self.include_subdirectories_var.get(),
            )
        finally:
            # Always restore the original method, even if an exception occurs
            captioning_model.caption_image = original_caption_image

        # Reload the current image data using image_handler
        if hasattr(self.parent, "image_handler") and hasattr(
            self.parent.image_handler, "load_image_data"
        ):
            self.parent.image_handler.load_image_data()
            self.parent.refresh_ui()
