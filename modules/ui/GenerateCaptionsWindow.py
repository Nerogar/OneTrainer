import csv
import os
import re
from tkinter import filedialog
from typing import Any

import customtkinter as ctk


class GenerateCaptionsWindow(ctk.CTkToplevel):
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

        self._setup_window("Batch generate captions", "400x540")
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

        self._create_layout(path, parent_include_subdirectories)

    def _setup_window(self, title: str, geometry: str) -> None:
        self.title(title)
        self.geometry(geometry)
        self.minsize(400, 380)
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
        self.additional_options_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.progress_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.buttons_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")

        self.basic_options_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=5)
        self.caption_options_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=5)
        self.additional_options_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=5)
        self.progress_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=5)
        self.buttons_frame.grid(row=4, column=0, columnspan=2, sticky="ew", pady=5)

        for frame in [
            self.basic_options_frame,
            self.caption_options_frame,
            self.additional_options_frame,
            self.progress_frame,
            self.buttons_frame,
        ]:
            self._configure_frame(frame)

        self._create_basic_options(path)
        self._create_caption_configuration()
        self._create_threshold_configuration()
        self._create_additional_options(include_subdirectories)
        self._create_progress_indicators()
        self._create_action_buttons()

        self.config_state["model_var"].trace_add("write", lambda *args: self._update_threshold_visibility())
        self._update_threshold_visibility()

    def _configure_frame(self, frame: ctk.CTkFrame) -> None:
        frame.grid_columnconfigure(0, weight=0, minsize=120)
        frame.grid_columnconfigure(1, weight=1)

    def _create_labeled_widget(
        self,
        parent: ctk.CTkFrame,
        label_text: str,
        widget: Any,
        row: int,
        widget_options: dict[str, Any] | None = None,
    ) -> None:
        label = ctk.CTkLabel(parent, text=label_text, anchor="w")
        label.grid(row=row, column=0, sticky="w", padx=(10, 5), pady=5)
        options = {"row": row, "column": 1, "sticky": "ew", "padx": (5, 10), "pady": 5}
        if widget_options:
            options.update(widget_options)
        widget.grid(**options)

    def _create_basic_options(self, path: str) -> None:
        self.model_dropdown = ctk.CTkOptionMenu(
            self.basic_options_frame,
            variable=self.config_state["model_var"],
            values=self.config_state["models"],
            dynamic_resizing=False,
            width=220,
        )
        self._create_labeled_widget(self.basic_options_frame, "Model", self.model_dropdown, row=0)

        path_frame = ctk.CTkFrame(self.basic_options_frame, fg_color="transparent")
        path_frame.grid(row=1, column=1, sticky="ew", padx=(5, 10), pady=5)
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
        path_label = ctk.CTkLabel(self.basic_options_frame, text="Folder", anchor="w")
        path_label.grid(row=1, column=0, sticky="w", padx=(10, 5), pady=5)

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
        self._create_labeled_widget(self.caption_options_frame, "Caption Postfix", self.postfix_entry, row=2)

        blacklist_frame = ctk.CTkFrame(self.caption_options_frame, fg_color="transparent")
        blacklist_frame.grid(row=3, column=1, sticky="ew", padx=(5, 10), pady=5)
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
        blacklist_label = ctk.CTkLabel(self.caption_options_frame, text="Blacklist", anchor="w")
        blacklist_label.grid(row=3, column=0, sticky="w", padx=(10, 5), pady=5)

        self.regex_enabled_var = ctk.BooleanVar(self, False)
        self.regex_enabled_checkbox = ctk.CTkCheckBox(
            self.caption_options_frame,
            text="Enable regex matching for blacklist",
            variable=self.regex_enabled_var,
        )
        self.regex_enabled_checkbox.grid(row=4, column=1, sticky="w", padx=(5, 10), pady=5)

        help_text = ("Enter tags to blacklist, separated by commas. You can also specify a .txt or .csv file path.")
        self.blacklist_help_label = ctk.CTkLabel(
            self.caption_options_frame,
            text=help_text,
            font=("", 12),
            wraplength=350,
            justify="left",
        )
        self.blacklist_help_label.grid(row=5, column=0, columnspan=2, sticky="w", padx=10, pady=5)

    def _create_threshold_configuration(self) -> None:
        self.threshold_frame.grid_columnconfigure(0, weight=0, minsize=120)
        self.threshold_frame.grid_columnconfigure(1, weight=1, minsize=60)
        self.threshold_frame.grid_columnconfigure(2, weight=0)

        self.general_threshold_label = ctk.CTkLabel(self.threshold_frame, text="General Tag Threshold", anchor="w")
        self.general_threshold_label.grid(row=0, column=0, sticky="w", padx=(10, 5), pady=5)
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

        self.character_threshold_label = ctk.CTkLabel(
            self.threshold_frame, text="Character Tag Threshold", anchor="w"
        )
        self.character_threshold_label.grid(row=1, column=0, sticky="w", padx=(10, 5), pady=5)
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

        explanation = (
            "MCut automatically determines optimal thresholds by finding the largest gap between adjacent "
            "label relevance scores. Enabling it disables the threshold inputs."
        )
        self.mcut_explanation_label = ctk.CTkLabel(
            self.threshold_frame,
            text=explanation,
            font=("", 12),
            wraplength=330,
            justify="left",
        )
        self.mcut_explanation_label.grid(row=2, column=0, columnspan=3, sticky="w", padx=10, pady=10)
        self._update_threshold_states()

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

    def _update_threshold_visibility(self, *args: Any) -> None:
        model = self.config_state["model_var"].get()
        threshold_height = 150
        is_supported_wd_model = any(name in model for name in ["WD SwinV2", "WD EVA02", "WD14 VIT"])

        if not self.general_mcut_var.get():
            if "EVA02" in model:
                self.general_threshold_var.set("0.5")
            elif is_supported_wd_model:
                self.general_threshold_var.set("0.35")

        geometry = self.geometry().split("+")[0]
        current_width, current_height = map(int, geometry.split("x"))
        if is_supported_wd_model and not self._threshold_visible:
            new_height = current_height + threshold_height
            self.threshold_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=5)
            self.additional_options_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=5)
            self.progress_frame.grid(row=4, column=0, columnspan=2, sticky="ew", pady=5)
            self.buttons_frame.grid(row=5, column=0, columnspan=2, sticky="ew", pady=5)
            self.geometry(f"{current_width}x{new_height}")
            self._threshold_visible = True
        elif not is_supported_wd_model and self._threshold_visible:
            new_height = current_height - threshold_height
            self.threshold_frame.grid_remove()
            self.additional_options_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=5)
            self.progress_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=5)
            self.buttons_frame.grid(row=4, column=0, columnspan=2, sticky="ew", pady=5)
            self.geometry(f"{current_width}x{new_height}")
            self._threshold_visible = False

    def _update_threshold_states(self) -> None:
        if self.general_mcut_var.get():
            self.general_threshold_var.set("0.0")
            self.general_threshold_entry.configure(state="disabled", placeholder_text="Auto")
            self.general_threshold_label.configure(text_color=("gray", "gray"))
        else:
            model = self.config_state["model_var"].get()
            self.general_threshold_var.set("0.5" if "EVA02" in model else "0.35")
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

    def get_blacklist_tags(self, blacklist_text: str, model_name: str) -> list[str]:
        """Convert blacklist_text to list depending on whether it's a file or a comma-separated string."""
        delimiter = "," if "WD" in model_name else None
        if blacklist_text.endswith(".txt") and os.path.isfile(blacklist_text):
            with open(blacklist_text, encoding="utf-8") as blacklist_file:
                return [line.rstrip("\n") for line in blacklist_file]
        elif blacklist_text.endswith(".csv") and os.path.isfile(blacklist_text):
            with open(blacklist_text, "r", encoding="utf-8") as blacklist_file:
                return [row[0] for row in csv.reader(blacklist_file)]
        elif delimiter:
            return [tag.strip() for tag in blacklist_text.split(delimiter)]
        else:
            return [blacklist_text]

    def parse_regex_blacklist(self, blacklist_tags: list[str], caption_tags: list[str]) -> list[str]:
        """Match regex patterns in blacklist against caption tags."""
        matched_tags: list[str] = []
        regex_spchars = set(".^$*+?!{}[]|()\\")
        for pattern in blacklist_tags:
            if any(char in regex_spchars for char in pattern):
                try:
                    compiled = re.compile(pattern, re.IGNORECASE)
                    for tag in caption_tags:
                        if compiled.fullmatch(tag) and tag not in matched_tags:
                            matched_tags.append(tag)
                except re.error:
                    pass
            else:
                pattern_lower = pattern.lower()
                for tag in caption_tags:
                    if tag.lower() == pattern_lower and tag not in matched_tags:
                        matched_tags.append(tag)
        return matched_tags

    def filter_blacklisted_tags(self, caption: str, model_name: str) -> str:
        blacklist_text = self.blacklist_entry.get().strip()
        if not blacklist_text:
            return caption

        # Check if the model contains "WD" and choose delimiter and joiner accordingly.
        if "WD" in model_name:
            delimiter = ","
            joiner = ", "
        else:
            delimiter = " "
            joiner = " "

        blacklist_tags = self.get_blacklist_tags(blacklist_text, model_name)
        # Trim each tag from the caption by splitting on the chosen delimiter.
        caption_tags = [tag.strip() for tag in caption.split(delimiter)]

        if self.regex_enabled_var.get():
            tags_to_remove = self.parse_regex_blacklist(blacklist_tags, caption_tags)
        else:
            tags_to_remove = []
            for tag in caption_tags:
                for blacklist_tag in blacklist_tags:
                    if tag.lower() == blacklist_tag.lower().strip():
                        tags_to_remove.append(tag)
                        break

        filtered_tags = [tag for tag in caption_tags if tag not in tags_to_remove]
        return joiner.join(filtered_tags)

    def set_progress(self, value: int, max_value: int) -> None:
        progress = value / max_value if max_value > 0 else 0
        self.progress.set(progress)
        self.progress_label.configure(text=f"{value}/{max_value}")
        self.progress.update()

    def create_captions(self) -> None:
        model_name = self.config_state["model_var"].get()
        wd_model_kwargs: dict[str, Any] = {}
        if "WD" in model_name:
            try:
                min_general = 0.0 if self.general_mcut_var.get() else float(self.general_threshold_var.get())
                wd_model_kwargs = {
                    "use_mcut_general": self.general_mcut_var.get(),
                    "use_mcut_character": self.character_mcut_var.get(),
                    "min_general_threshold": min_general,
                    "min_character_threshold": 0.15,
                }
            except ValueError:
                print("Invalid threshold value, using defaults")
                wd_model_kwargs = {
                    "use_mcut_general": self.general_mcut_var.get(),
                    "use_mcut_character": self.character_mcut_var.get(),
                }

        if "WD" in model_name:
            self.parent.model_manager.captioning_model = None
            self.parent.model_manager.current_captioning_model_name = None
            model_class = self.parent.model_manager._captioning_registry[model_name]
            captioning_model = model_class(
                self.parent.model_manager.device,
                self.parent.model_manager.precision,
                model_name=model_name,
                **wd_model_kwargs,
            )
            self.parent.model_manager.captioning_model = captioning_model
            self.parent.model_manager.current_captioning_model_name = model_name
        else:
            captioning_model = self.parent.model_manager.load_captioning_model(model_name)

        if captioning_model is None:
            print(f"Failed to load model: {model_name}")
            return

        # Get the blacklist once at the beginning to avoid caching issues
        blacklist_text = self.blacklist_entry.get().strip()
        blacklist_tags = []
        if blacklist_text:
            blacklist_tags = self.get_blacklist_tags(blacklist_text, model_name)
            print(f"Loaded blacklist tags: {blacklist_tags}")

        # Store the original generate_caption method
        original_generate_caption = captioning_model.generate_caption

        def generate_caption_with_blacklist(caption_sample, initial_caption, caption_prefix, caption_postfix):
            caption = original_generate_caption(caption_sample, initial_caption, caption_prefix, caption_postfix)
            filtered_caption = self.filter_blacklisted_tags(caption, model_name)
            print(f"Original caption: {caption}", flush=True)
            print(f"Filtered caption: {filtered_caption}", flush=True)
            return filtered_caption

        # Replace generate_caption with our wrapper
        captioning_model.generate_caption = generate_caption_with_blacklist

        mode = self.config_state["mode_mapping"][self.config_state["mode_var"].get()]
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
            # Restore the original generate_caption method
            captioning_model.generate_caption = original_generate_caption

        if hasattr(self.parent, "image_handler") and hasattr(self.parent.image_handler, "load_image_data"):
            self.parent.image_handler.load_image_data()
            self.parent.refresh_ui()
