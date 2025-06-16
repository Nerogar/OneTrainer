import logging
import os
import threading
from tkinter import TclError, filedialog, messagebox
from typing import Any, TypedDict

from modules.module.captioning.caption_config_types import (
    BaseGenerationConfig,
    BlipGenerationConfig,
    JoyCaptionGenerationConfig,
    MoondreamGenerationConfig,
    WDGenerationConfig,
)
from modules.module.captioning.captioning_util import (
    filter_blacklisted_tags,
)
from modules.module.captioning.CaptionSample import CaptionSample
from modules.module.JoyCaptionModel import (
    CAPTION_LENGTH_CHOICES,
    CAPTION_TYPE_MAP,
    EXTRA_OPTIONS,
    NAME_OPTION,
)
from modules.module.JoyCaptionModel import (
    DEFAULT_MAX_TOKENS as JOY_DEFAULT_MAX_TOKENS,
)
from modules.util.ui.ui_utils import (
    set_window_icon,
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
    SCROLLABLE_FRAME_LABEL_WRAPLENGTH: int = 300 # Adjusted for checkboxes in scrollable frame

    APP_MIN_WIDTH: int = 390
    APP_MIN_HEIGHT: int = 510

    # Defaults for JoyCaption Advanced Options
    JOY_DEFAULT_TEMPERATURE: float = 0.6
    JOY_DEFAULT_TOP_P: float = 0.9
    JOY_DEFAULT_TOP_K: int = 0
    # JOY_DEFAULT_MAX_TOKENS is already imported
    JOY_DEFAULT_REPETITION_PENALTY: float = 1.2
    JOY_DEFAULT_GPU_INDEX: int = 0

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
        self._joycaption_visible: bool = False
        self._joy_name_input_visible: bool = False
        self._joycaption_width_applied: bool = False
        self._joy_advanced_toggle_visible: bool = False # New flag
        self._joy_advanced_options_visible: bool = False # New flag

        self._setup_window("Batch generate captions", "405x520") #initial dim
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # 1) Populate basic config_state
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

        # 2) Add caption length options for Moondream2
        self.config_state["caption_lengths"] = ["short", "normal", "long"]
        self.config_state["caption_length_var"] = ctk.StringVar(self, "normal")

        # JoyCaption Advanced Mode Variable
        self.joy_advanced_mode_var = ctk.BooleanVar(self, False)
        self.joy_advanced_mode_var.trace_add("write", lambda *args: self._update_model_specific_options())

        # 3) Build the layout (this also creates the new JoyCaption frame)
        self._create_layout(path, parent_include_subdirectories)

    def _setup_window(self, title: str, geometry: str) -> None:
        self.title(title)
        self.geometry(geometry)
        self.minsize(self.APP_MIN_WIDTH, self.APP_MIN_HEIGHT)
        self.resizable(True, True)
        self.wait_visibility()
        self.grab_set()
        self.focus_set()
        self.after(200, lambda: set_window_icon(self))

    def _create_layout(self, path: str, include_subdirectories: bool) -> None:
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        # Configure columns for main_frame itself.
        # Sub-frames are typically added to main_frame at column=0 with columnspan=2,
        # meaning they occupy main_frame's columns 0 and 1.
        # Column 0 of main_frame will effectively hold the labels from the sub-frames.
        # Column 1 of main_frame will effectively hold the widgets from the sub-frames.
        self.main_frame.grid_columnconfigure(0, weight=0, minsize=120) # Corresponds to sub-frame label column width
        self.main_frame.grid_columnconfigure(1, weight=1)             # Corresponds to sub-frame widget column, should expand

        # Create each sub‐frame
        self.basic_options_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.caption_options_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.joycaption_options_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.threshold_frame = ctk.CTkFrame(self.main_frame)
        self.moondream_options_frame = ctk.CTkFrame(self.main_frame)
        self.additional_options_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.joy_advanced_toggle_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent") # New frame
        self.joy_advanced_options_frame = ctk.CTkFrame(self.main_frame) # New frame
        self.progress_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.buttons_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")

        # Put the first two frames in a fixed vertical order;
        # the joycaption, threshold, and moondream frames will be shown/hidden dynamically
        frames = [
            self.basic_options_frame,
            self.caption_options_frame,
            # ‼ We do not grid the joycaption/threshold/moondream frames here; they are placed by _update_model_specific_options()
            # ‼ We also do not grid joy_advanced_toggle_frame or joy_advanced_options_frame here
            self.additional_options_frame,
            self.progress_frame,
            self.buttons_frame,
        ]

        for i, frame in enumerate(frames):
            frame.grid(row=i, column=0, columnspan=2, sticky="ew", pady=self.DEFAULT_PADY)
            self._configure_standard_frame(frame)

        # Make sure each "optional" frame is at least configured;
        # they will only be gridded when needed
        self._configure_standard_frame(self.threshold_frame)
        self._configure_standard_frame(self.moondream_options_frame)
        self._configure_standard_frame(self.joycaption_options_frame)
        self._configure_standard_frame(self.joy_advanced_toggle_frame) # Configure new frame
        self._configure_standard_frame(self.joy_advanced_options_frame) # Configure new frame

        # Now build each section
        self._create_basic_options(path)
        self._create_caption_configuration()
        self._create_joycaption_configuration()
        self._create_threshold_configuration()
        self._create_moondream_configuration()
        self._create_additional_options(include_subdirectories)
        self._create_joycaption_advanced_toggle() # New
        self._create_joycaption_advanced_options() # New
        self._create_progress_indicators()
        self._create_action_buttons()

        # Whenever the model selection changes, toggle the right frames on/off
        self.config_state["model_var"].trace_add("write", lambda *args: self._update_model_specific_options())
        self._update_model_specific_options()

    def _configure_standard_frame(self, frame: ctk.CTkFrame) -> None:
        """Configure a frame with two columns: label column (fixed) + widget column (expand)."""
        frame.grid_columnconfigure(0, weight=0, minsize=120)
        frame.grid_columnconfigure(1, weight=1)

    def _create_labeled_widget(
        self,
        parent: ctk.CTkFrame,
        label_text: str,
        widget: Any,
        row: int,
        widget_options: GridOptions | None = None,
        label_options: GridOptions | None = None,
    ) -> None:
        """Create a label in column 0 and the given widget in column 1 on the same row."""
        label_grid = self.LABEL_GRID.copy()
        label_grid["row"] = row
        label_grid["column"] = 0
        if label_options:
            label_grid.update(label_options)

        label = ctk.CTkLabel(parent, text=label_text, anchor="w")
        label.grid(**label_grid)

        widget_grid = self.WIDGET_GRID.copy()
        widget_grid["row"] = row
        widget_grid["column"] = 1
        if widget_options:
            widget_grid.update(widget_options)

        widget.grid(**widget_grid)

    def _create_explanation_label(
        self,
        parent: ctk.CTkFrame,
        text: str,
        row: int,
        columnspan: int = 2,
        wraplength: int | None = None,
    ) -> ctk.CTkLabel:
        """Create a standardized explanation label (smaller font, wrapped)."""
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

    # ─── Basic options: folder, model, mode ─────────────────────────────────
    def _create_basic_options(self, path: str) -> None:
        # Model dropdown
        self.model_dropdown = ctk.CTkOptionMenu(
            self.basic_options_frame,
            variable=self.config_state["model_var"],
            values=self.config_state["models"],
            dynamic_resizing=False,
            width=220,
        )
        self._create_labeled_widget(self.basic_options_frame, "Model", self.model_dropdown, row=0)

        # Path selection with [ ... ] button
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

    # ─── Common caption‐related options (initial/prefix/postfix/blacklist) ─────
    def _create_caption_configuration(self) -> None:
        # Row 0: Initial Caption (handled specially to enable/disable its label and entry)
        self.initial_caption_label = ctk.CTkLabel(self.caption_options_frame, text="Initial Caption", anchor="w")
        self.initial_caption_label.grid(row=0, column=0, **self.LABEL_GRID)
        self.caption_entry = ctk.CTkEntry(self.caption_options_frame)
        self.caption_entry.grid(row=0, column=1, **self.WIDGET_GRID)

        # Row 1: Caption Prefix
        self.prefix_label = ctk.CTkLabel(self.caption_options_frame, text="Caption Prefix", anchor="w")
        self.prefix_label.grid(row=1, column=0, **self.LABEL_GRID)
        self.prefix_entry = ctk.CTkEntry(self.caption_options_frame)
        self.prefix_entry.grid(row=1, column=1, **self.WIDGET_GRID)

        # Row 2: Caption Suffix
        self.postfix_label = ctk.CTkLabel(self.caption_options_frame, text="Caption Suffix", anchor="w")
        self.postfix_label.grid(row=2, column=0, **self.LABEL_GRID)
        self.postfix_entry = ctk.CTkEntry(self.caption_options_frame)
        self.postfix_entry.grid(row=2, column=1, **self.WIDGET_GRID)

        # Row 3: Blacklist
        blacklist_label = ctk.CTkLabel(self.caption_options_frame, text="Blacklist", anchor="w")
        blacklist_label.grid(row=3, column=0, **self.LABEL_GRID)

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

        # Row 4: Regex checkbox
        self.regex_enabled_var = ctk.BooleanVar(self, False)
        self.regex_enabled_checkbox = ctk.CTkCheckBox(
            self.caption_options_frame,
            text="Enable regex matching for blacklist",
            variable=self.regex_enabled_var,
        )
        widget_grid_no_sticky = {k: v for k, v in self.WIDGET_GRID.items() if k != 'sticky'}
        self.regex_enabled_checkbox.grid(row=4, column=1, sticky="w", **widget_grid_no_sticky)

        # Row 5: Help text for blacklist
        help_text = "Enter tags to blacklist, separated by commas, or load a .txt/.csv file."
        self._create_explanation_label(self.caption_options_frame, help_text, row=5)

    # ─── JoyCaption‐specific UI ───────────────────────────────────────────────
    def _create_joycaption_configuration(self) -> None:
        """
        These controls appear only when "JoyCaption" is selected:
         - Caption Type (one of CAPTION_TYPE_MAP keys)
         - Caption Length (one of CAPTION_LENGTH_CHOICES)
         - Name (for NAME_OPTION) - conditionally visible
         - Extra Options (scrollable list of checkboxes with wrapped labels)
         - Live Prompt Preview (Textbox)
         - Temperature, Top-p, Max Tokens
        """
        # Create all widgets first. Their gridding will be handled by _layout_joycaption_widgets.
        self.joy_caption_type_label = ctk.CTkLabel(self.joycaption_options_frame, text="Caption Type", anchor="w")
        self.joy_caption_type_var = ctk.StringVar(self, list(CAPTION_TYPE_MAP.keys())[0])
        self.joy_caption_type_var.trace_add("write", self._update_joycaption_prompt_display)
        self.joy_caption_type_dropdown = ctk.CTkOptionMenu(
            self.joycaption_options_frame,
            variable=self.joy_caption_type_var,
            values=list(CAPTION_TYPE_MAP.keys()),
            dynamic_resizing=False,
            width=200,
        )

        self.joy_caption_length_label = ctk.CTkLabel(self.joycaption_options_frame, text="Caption Length", anchor="w")
        self.joy_caption_length_var = ctk.StringVar(self, CAPTION_LENGTH_CHOICES[0])
        self.joy_caption_length_var.trace_add("write", self._update_joycaption_prompt_display)
        self.joy_caption_length_dropdown = ctk.CTkOptionMenu(
            self.joycaption_options_frame,
            variable=self.joy_caption_length_var,
            values=CAPTION_LENGTH_CHOICES,
            dynamic_resizing=False,
            width=120,
        )

        # --- Name Input (conditionally visible) ---
        self.joy_name_input_label = ctk.CTkLabel(self.joycaption_options_frame, text="Name", anchor="w") # Simpler label
        self.joy_name_var = ctk.StringVar(self)
        self.joy_name_var.trace_add("write", self._update_joycaption_prompt_display)
        self.joy_name_entry = ctk.CTkEntry(self.joycaption_options_frame, textvariable=self.joy_name_var)

        # --- Extra Options Label (Above) ---
        self.joy_extra_options_header_label = ctk.CTkLabel(self.joycaption_options_frame, text="Extra Options:", anchor="w")

        # --- Extra Options (Scrollable Checkboxes with Wrapped Labels) ---
        self.joy_extra_options_scrollable_frame = ctk.CTkScrollableFrame(
            self.joycaption_options_frame,
            height=150
        )
        self.joy_extra_options_scrollable_frame.grid_columnconfigure(0, weight=1)

        self.joy_extra_options_checkboxes: list[tuple[ctk.BooleanVar, str]] = []
        inner_label_wraplength = self.SCROLLABLE_FRAME_LABEL_WRAPLENGTH

        for option_text in EXTRA_OPTIONS:
            option_item_frame = ctk.CTkFrame(self.joy_extra_options_scrollable_frame, fg_color="transparent")
            option_item_frame.pack(fill="x", expand=True, pady=1, padx=1)
            option_item_frame.grid_columnconfigure(0, weight=0)
            option_item_frame.grid_columnconfigure(1, weight=1)

            var = ctk.BooleanVar(self, False)
            var.trace_add("write", self._update_joycaption_prompt_display)

            if option_text == NAME_OPTION:
                self.joy_name_option_active_var = var # Store the specific var for NAME_OPTION
                # Trace this var to update the visibility of the name input field
                var.trace_add("write", self._on_name_option_toggled)

            checkbox = ctk.CTkCheckBox(option_item_frame, text="", variable=var, width=20)
            checkbox.grid(row=0, column=0, sticky="nw", padx=(0, 5))
            option_label = ctk.CTkLabel(
                option_item_frame, text=option_text, wraplength=inner_label_wraplength, justify="left", anchor="w"
            )
            option_label.grid(row=0, column=1, sticky="ew")
            self.joy_extra_options_checkboxes.append((var, option_text))

        # --- Live Prompt Preview Textbox ---
        self.joy_prompt_preview_label = ctk.CTkLabel(self.joycaption_options_frame, text="Live Prompt Preview", anchor="w")
        self.joy_live_prompt_textbox = ctk.CTkTextbox(
            self.joycaption_options_frame, height=100, wrap="word"
        )

        # --- Temperature, Top-p, Max Tokens ---
        # self.joy_temperature_label = ctk.CTkLabel(self.joycaption_options_frame, text="Temperature (0.0–1.0)", anchor="w")
        # self.joy_temperature_var = ctk.StringVar(self, str(JoyCaptionModel.DEFAULT_TEMPERATURE) if hasattr(JoyCaptionModel, "DEFAULT_TEMPERATURE") else "0.6")
        # self.joy_temperature_entry = ctk.CTkEntry(self.joycaption_options_frame, textvariable=self.joy_temperature_var)

        # self.joy_topp_label = ctk.CTkLabel(self.joycaption_options_frame, text="Top-p (0.0–1.0)", anchor="w")
        # self.joy_topp_var = ctk.StringVar(self, str(JoyCaptionModel.DEFAULT_TOP_P) if hasattr(JoyCaptionModel, "DEFAULT_TOP_P") else "0.9")
        # self.joy_topp_entry = ctk.CTkEntry(self.joycaption_options_frame, textvariable=self.joy_topp_var)

        # self.joy_max_tokens_label = ctk.CTkLabel(self.joycaption_options_frame, text="Max Tokens", anchor="w")
        # self.joy_max_tokens_var = ctk.StringVar(self, str(JOY_DEFAULT_MAX_TOKENS))
        # self.joy_max_tokens_entry = ctk.CTkEntry(self.joycaption_options_frame, textvariable=self.joy_max_tokens_var)

        # Initial layout of JoyCaption widgets
        self._layout_joycaption_widgets()
        # Initial prompt update
        self._update_joycaption_prompt_display()

    def _on_name_option_toggled(self, *args: Any) -> None:
        """Called when the NAME_OPTION checkbox state changes."""
        self._layout_joycaption_widgets()
        self._update_joycaption_prompt_display() # Ensure prompt updates immediately

    def _layout_joycaption_widgets(self) -> None:
        """Grids or removes widgets within the joycaption_options_frame based on current states."""
        current_row = 0

        # --- Caption Type ---
        self.joy_caption_type_label.grid(row=current_row, column=0, **self.LABEL_GRID)
        self.joy_caption_type_dropdown.grid(row=current_row, column=1, **self.WIDGET_GRID)
        current_row += 1

        # --- Caption Length ---
        self.joy_caption_length_label.grid(row=current_row, column=0, **self.LABEL_GRID)
        self.joy_caption_length_dropdown.grid(row=current_row, column=1, **self.WIDGET_GRID)
        current_row += 1

        # --- Name Input (conditionally visible) ---
        if hasattr(self, 'joy_name_option_active_var') and self.joy_name_option_active_var.get():
            self.joy_name_input_label.grid(row=current_row, column=0, **self.LABEL_GRID)
            self.joy_name_entry.grid(row=current_row, column=1, **self.WIDGET_GRID)
            self._joy_name_input_visible = True
            current_row += 1
        else:
            self.joy_name_input_label.grid_remove()
            self.joy_name_entry.grid_remove()
            if hasattr(self, 'joy_name_var'): # Clear the variable if field is hidden
                self.joy_name_var.set("")
            self._joy_name_input_visible = False


        # --- Extra Options Label ---
        self.joy_extra_options_header_label.grid(
            row=current_row, column=0, columnspan=2, sticky="w",
            padx=self.DEFAULT_PADX[0], pady=(self.DEFAULT_PADY, 2)
        )
        current_row += 1

        # --- Extra Options Scrollable Frame ---
        self.joy_extra_options_scrollable_frame.grid(
            row=current_row, column=0, columnspan=2, sticky="nsew",
            padx=self.DEFAULT_PADX, pady=(0, self.DEFAULT_PADY)
        )
        current_row += 1

        # --- Live Prompt Preview ---
        self.joy_prompt_preview_label.grid(
            row=current_row, column=0, columnspan=2, sticky="w",
            padx=self.DEFAULT_PADX[0], pady=(self.DEFAULT_PADY, 2) # Position above textbox
        )
        current_row += 1
        self.joy_live_prompt_textbox.grid(
            row=current_row, column=0, columnspan=2, sticky="ew", # Span both columns
            padx=self.DEFAULT_PADX, pady=(0, self.DEFAULT_PADY)
        )
        current_row += 1

        # --- Temperature ---
        # self.joy_temperature_label.grid(row=current_row, column=0, **self.LABEL_GRID)
        # self.joy_temperature_entry.grid(row=current_row, column=1, **self.WIDGET_GRID)
        # current_row += 1

        # # --- Top‐p ---
        # self.joy_topp_label.grid(row=current_row, column=0, **self.LABEL_GRID)
        # self.joy_topp_entry.grid(row=current_row, column=1, **self.WIDGET_GRID)
        # current_row += 1

        # # --- Max Tokens ---
        # self.joy_max_tokens_label.grid(row=current_row, column=0, **self.LABEL_GRID)
        # self.joy_max_tokens_entry.grid(row=current_row, column=1, **self.WIDGET_GRID)
        # current_row += 1

    def _update_joycaption_prompt_display(self, *args: Any) -> None:
        """Constructs and displays the JoyCaption prompt based on UI selections."""
        if not hasattr(self, 'joy_caption_type_var') or not self.joycaption_options_frame.winfo_ismapped():
            return

        caption_type_key = self.joy_caption_type_var.get()
        caption_length_str = self.joy_caption_length_var.get()
        # Name input is now sourced conditionally based on its own checkbox

        prompt_parts: list[str] = []

        # 1. Base template from Caption Type and Length
        if caption_type_key in CAPTION_TYPE_MAP:
            prompt_templates = CAPTION_TYPE_MAP[caption_type_key]
            chosen_template: str
            if caption_length_str == "any":
                chosen_template = prompt_templates[0]
            elif caption_length_str.isdigit():
                chosen_template = prompt_templates[1].replace("{word_count}", caption_length_str)
            else: # 'short', 'long', etc.
                chosen_template = prompt_templates[2].replace("{length}", caption_length_str)
            prompt_parts.append(chosen_template)

        # 2. Add selected Extra Options
        if hasattr(self, 'joy_extra_options_checkboxes'):
            for var, option_text in self.joy_extra_options_checkboxes:
                if var.get(): # If checkbox is selected
                    if option_text == NAME_OPTION:
                        # Check if the NAME_OPTION checkbox (self.joy_name_option_active_var) is ticked
                        if hasattr(self, 'joy_name_option_active_var') and self.joy_name_option_active_var.get():
                            name_value_from_entry = self.joy_name_var.get().strip()
                            if name_value_from_entry: # Only add if the name entry is filled
                                formatted_name_option = NAME_OPTION.replace("{name}", name_value_from_entry)
                                prompt_parts.append(formatted_name_option)
                    else:
                        # For other extra options, add their text directly
                        prompt_parts.append(option_text)

        final_prompt = " ".join(prompt_parts).strip().replace("  ", " ")

        if hasattr(self, 'joy_live_prompt_textbox'):
            # Preserve cursor and selection if possible (might be tricky with full replace)
            # For simplicity, just replace content.
            current_content = self.joy_live_prompt_textbox.get("1.0", "end-1c")
            if current_content != final_prompt: # Update only if changed to avoid unnecessary flicker/cursor jumps
                self.joy_live_prompt_textbox.delete("1.0", "end")
                self.joy_live_prompt_textbox.insert("1.0", final_prompt)

    # ─── Threshold configuration (for WD models) ───────────────────────────────
    def _create_threshold_configuration(self) -> None:
        self.threshold_frame.grid_columnconfigure(0, weight=0, minsize=120)
        self.threshold_frame.grid_columnconfigure(1, weight=1, minsize=60)
        self.threshold_frame.grid_columnconfigure(2, weight=0)

        # General threshold
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

        # Character threshold
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

        explanation = (
            "MCut automatically finds the largest gap between adjacent label relevance scores. "
            "Enabling it disables manual threshold inputs."
        )
        self._create_explanation_label(self.threshold_frame, explanation, row=2, columnspan=3)

        self._update_threshold_states()

    # ─── Moondream2 configuration ─────────────────────────────────────
    def _create_moondream_configuration(self) -> None:
        # Caption length dropdown (short vs. normal)
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

        explanation = "Choose 'short' for a brief description or 'normal' for a more detailed caption and 'long' for an even more detailed caption."
        self._create_explanation_label(self.moondream_options_frame, explanation, row=1)

    # ─── “Include subfolders” switch ───────────────────────────────────────────
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

    # ─── JoyCaption Advanced Mode Toggle ──────────────────────────────────────
    def _create_joycaption_advanced_toggle(self) -> None:
        """Creates the switch to toggle JoyCaption advanced options."""
        self.joy_advanced_label = ctk.CTkLabel(
            self.joy_advanced_toggle_frame, text="Advanced JoyCaption Options", anchor="w"
        )
        # self.joy_advanced_label.grid(row=0, column=0, sticky="w", padx=(10, 5), pady=5) # Gridded by _create_labeled_widget

        self.joy_advanced_switch = ctk.CTkSwitch(
            self.joy_advanced_toggle_frame, text="", variable=self.joy_advanced_mode_var
        )
        # self.joy_advanced_switch.grid(row=0, column=1, sticky="w", padx=(5, 10), pady=5) # Gridded by _create_labeled_widget
        self._create_labeled_widget(
            self.joy_advanced_toggle_frame,
            "Advanced JoyCaption Options",
            self.joy_advanced_switch,
            row=0,
            label_options={"sticky": "w", "padx": (10,5), "pady": self.DEFAULT_PADY},
            widget_options={"sticky": "w", "padx": (5,10), "pady": self.DEFAULT_PADY}
        )

    # ─── JoyCaption Advanced Options UI ───────────────────────────────────────
    def _create_joycaption_advanced_options(self) -> None:
        """Creates the input fields for advanced JoyCaption parameters."""
        current_row = 0

        # Temperature
        self.joy_temp_var = ctk.StringVar(self, str(self.JOY_DEFAULT_TEMPERATURE))
        self.joy_temp_entry = ctk.CTkEntry(self.joy_advanced_options_frame, textvariable=self.joy_temp_var, width=70)
        self._create_labeled_widget(self.joy_advanced_options_frame, "Temperature (0-1)", self.joy_temp_entry, row=current_row)
        current_row += 1

        # Top-p
        self.joy_topp_var = ctk.StringVar(self, str(self.JOY_DEFAULT_TOP_P))
        self.joy_topp_entry = ctk.CTkEntry(self.joy_advanced_options_frame, textvariable=self.joy_topp_var, width=70)
        self._create_labeled_widget(self.joy_advanced_options_frame, "Top-p (0-1)", self.joy_topp_entry, row=current_row)
        current_row += 1

        # Top-K
        self.joy_topk_var = ctk.StringVar(self, str(self.JOY_DEFAULT_TOP_K))
        self.joy_topk_entry = ctk.CTkEntry(self.joy_advanced_options_frame, textvariable=self.joy_topk_var, width=70)
        self._create_labeled_widget(self.joy_advanced_options_frame, "Top-K (int, 0 for none)", self.joy_topk_entry, row=current_row)
        current_row += 1

        # Max Tokens
        self.joy_max_tokens_var = ctk.StringVar(self, str(JOY_DEFAULT_MAX_TOKENS))
        self.joy_max_tokens_entry = ctk.CTkEntry(self.joy_advanced_options_frame, textvariable=self.joy_max_tokens_var, width=70)
        self._create_labeled_widget(self.joy_advanced_options_frame, "Max Tokens (int)", self.joy_max_tokens_entry, row=current_row)
        current_row += 1

        # Repetition Penalty
        self.joy_rep_penalty_var = ctk.StringVar(self, str(self.JOY_DEFAULT_REPETITION_PENALTY))
        self.joy_rep_penalty_entry = ctk.CTkEntry(self.joy_advanced_options_frame, textvariable=self.joy_rep_penalty_var, width=70)
        self._create_labeled_widget(self.joy_advanced_options_frame, "Repetition Penalty (float)", self.joy_rep_penalty_entry, row=current_row)
        current_row += 1

        # GPU Index
        self.joy_gpu_index_var = ctk.StringVar(self, str(self.JOY_DEFAULT_GPU_INDEX))
        self.joy_gpu_index_entry = ctk.CTkEntry(self.joy_advanced_options_frame, textvariable=self.joy_gpu_index_var, width=70)
        self._create_labeled_widget(self.joy_advanced_options_frame, "GPU Index (int, -1 auto)", self.joy_gpu_index_entry, row=current_row)
        current_row += 1

        explanation = "These settings provide finer control over JoyCaption generation. Invalid values may cause errors."
        self._create_explanation_label(self.joy_advanced_options_frame, explanation, row=current_row)


    # ─── Progress bar + label ──────────────────────────────────────────────────
    def _create_progress_indicators(self) -> None:
        self.progress_label = ctk.CTkLabel(self.progress_frame, text="Progress: 0/0", anchor="w")
        self.progress_label.grid(row=0, column=0, sticky="w", padx=(10, 5), pady=5)
        self.progress = ctk.CTkProgressBar(self.progress_frame, orientation="horizontal", mode="determinate")
        self.progress.grid(row=0, column=1, sticky="ew", padx=(5, 10), pady=5)
        self.progress.set(0)

    # ─── “Create Captions” button ───────────────────────────────────────────────
    def _create_action_buttons(self) -> None:
        self.create_captions_button = ctk.CTkButton(
            self.buttons_frame, text="Generate Captions", command=self.create_captions
        )
        self.create_captions_button.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=5)

    # ─── Browse helper ─────────────────────────────────────────────────────────
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

    # ─── Model‐type heuristics ───────────────────────────────────────────────────
    def _is_moondream_model(self, model_name: str) -> bool:
        """Check if the model is a Moondream model."""
        return "MOONDREAM" in model_name.upper()

    def _is_joycaption_model(self, model_name: str) -> bool:
        """Check if the model is exactly 'JoyCaption' (or starts with it)."""
        return model_name.lower().startswith("joycaption")

    def _is_blip_model(self, model_name: str) -> bool:
        """Check if the model is a BLIP model (e.g., BLIP2)."""
        return "BLIP" in model_name.upper()

    def _is_joytag_model(self, model_name: str) -> bool:
        """Check if the model is a JoyTag model."""
        return "JOYTAG" in model_name.upper()

    def _is_booru_model(self, model_name: str) -> bool:
        """Check if the model is a Booru-style model."""
        return "BOORU" in model_name.upper()

    def _is_wd_model(self, model_name: str) -> bool:
        """Check if the model is a WD-style (tagger) model, typically requiring threshold options."""
        # Adjust this logic if your WD/tagger model identification is different.
        # This includes JoyTag and Booru as they often use similar thresholding.
        return ("WD" in model_name.upper() or
                self._is_joytag_model(model_name) or
                self._is_booru_model(model_name))

    # ─── Show/hide Booru, Moondream, JoyCaption frames ────────────────────────────
    def _update_model_specific_options(self, *args: Any) -> None:
        """Update which option frames are visible and UI state based on the selected model."""
        model = self.config_state["model_var"].get()
        is_joy_model_selected = self._is_joycaption_model(model)

        # Update Initial Caption field state
        if hasattr(self, 'initial_caption_label') and hasattr(self, 'caption_entry'):
            if self._is_blip_model(model):
                self.initial_caption_label.grid(row=0, column=0, **self.LABEL_GRID)
                self.caption_entry.grid(row=0, column=1, **self.WIDGET_GRID)
                self.initial_caption_label.configure(text_color=ctk.ThemeManager.theme["CTkLabel"]["text_color"])
                self.caption_entry.configure(state="normal", placeholder_text="")
            else:
                self.initial_caption_label.grid_remove()
                self.caption_entry.grid_remove()
                self.caption_entry.delete(0, "end")

        # Update Prefix/Suffix fields state based on JoyCaption
        is_joy = self._is_joycaption_model(model)
        if hasattr(self, 'prefix_label') and hasattr(self, 'prefix_entry') and \
           hasattr(self, 'postfix_label') and hasattr(self, 'postfix_entry'):
            if is_joy:
                self.prefix_label.grid_remove()
                self.prefix_entry.grid_remove()
                self.prefix_entry.delete(0, "end")
                self.postfix_label.grid_remove()
                self.postfix_entry.grid_remove()
                self.postfix_entry.delete(0, "end")
            else:
                self.prefix_label.grid(row=1, column=0, **self.LABEL_GRID)
                self.prefix_entry.grid(row=1, column=1, **self.WIDGET_GRID)
                self.postfix_label.grid(row=2, column=0, **self.LABEL_GRID)
                self.postfix_entry.grid(row=2, column=1, **self.WIDGET_GRID)
                # Ensure they are enabled and colors are reset if they were previously changed
                self.prefix_entry.configure(state="normal")
                self.prefix_label.configure(text_color=ctk.ThemeManager.theme["CTkLabel"]["text_color"])
                self.postfix_entry.configure(state="normal")
                self.postfix_label.configure(text_color=ctk.ThemeManager.theme["CTkLabel"]["text_color"])


        # Get current window dimensions and position for resizing
        full_geometry_str = self.geometry() # e.g., "400x300+50+50" or "400x300"
        geom_parts = full_geometry_str.split('x')
        current_width = int(geom_parts[0])
        height_and_pos_parts = geom_parts[1].split('+')
        current_height = int(height_and_pos_parts[0])
        current_pos_str = ""
        if len(height_and_pos_parts) > 2: # Check if position data exists (+x+y)
            current_pos_str = f"+{height_and_pos_parts[1]}+{height_and_pos_parts[2]}"
        elif len(height_and_pos_parts) > 1: # Check for older format or incomplete position
             # This case might indicate an issue or a format like "WxH+X" which is unusual.
             # For safety, if only one '+' is found, it might be part of a non-standard geometry string.
             # We'll assume it's not a valid position string unless two parts after height exist.
             # If your system might produce "WxH+X", this logic might need adjustment.
             # However, standard Tk geometry is "widthxheight±x±y".
             pass


        new_width = current_width
        new_height = current_height # Start with current height, adjust based on frame visibility changes

        is_joy = self._is_joycaption_model(model)
        joy_width_adjustment = 10

        if is_joy:
            if not self._joycaption_width_applied:
                new_width += joy_width_adjustment
                self._joycaption_width_applied = True
        else: # Not JoyCaption
            if self._joycaption_width_applied:
                new_width -= joy_width_adjustment
                self._joycaption_width_applied = False

        # Ensure new_width is not less than min_width defined by the application
        new_width = max(new_width, self.APP_MIN_WIDTH)

        # --- Dynamic frames positioning ---
        # Base row index in main_frame for the first dynamic/optional frame
        current_main_frame_row = 2 # Starts after basic_options_frame and caption_options_frame

        # 1) WD threshold frame
        is_wd = self._is_wd_model(model)
        threshold_height_change = 110
        if is_wd:
            if not self._threshold_visible:
                new_height += threshold_height_change
            self.threshold_frame.grid(row=current_main_frame_row, column=0, columnspan=2, sticky="ew", pady=self.DEFAULT_PADY)
            self._threshold_visible = True
            current_main_frame_row += 1
        else:
            if self._threshold_visible:
                new_height -= threshold_height_change
            self.threshold_frame.grid_remove()
            self._threshold_visible = False

        # 2) Moondream frame
        is_moondream = self._is_moondream_model(model)
        moondream_height_change = 70
        if is_moondream:
            if not self._moondream_options_visible:
                new_height += moondream_height_change
            self.moondream_options_frame.grid(row=current_main_frame_row, column=0, columnspan=2, sticky="ew", pady=self.DEFAULT_PADY)
            self._moondream_options_visible = True
            current_main_frame_row += 1
        else:
            if self._moondream_options_visible:
                new_height -= moondream_height_change
            self.moondream_options_frame.grid_remove()
            self._moondream_options_visible = False

        # 3) JoyCaption main options frame
        joy_base_height = 410
        joy_name_field_height = 40
        joy_main_options_height_change = joy_base_height
        if hasattr(self, '_joy_name_input_visible') and self._joy_name_input_visible:
            joy_main_options_height_change += joy_name_field_height

        if is_joy_model_selected:
            if not self._joycaption_visible:
                new_height += joy_main_options_height_change
            self.joycaption_options_frame.grid(row=current_main_frame_row, column=0, columnspan=2, sticky="ew", pady=self.DEFAULT_PADY)
            self._joycaption_visible = True
            self._layout_joycaption_widgets()
            self._update_joycaption_prompt_display()
            current_main_frame_row += 1
        else:
            if self._joycaption_visible:
                new_height -= joy_main_options_height_change
            self.joycaption_options_frame.grid_remove()
            self._joycaption_visible = False

        # 4) Additional options frame (Include subdirectories) - always present after model-specifics
        self.additional_options_frame.grid(row=current_main_frame_row, column=0, columnspan=2, sticky="ew", pady=self.DEFAULT_PADY)
        current_main_frame_row += 1

        # 5) JoyCaption Advanced Toggle Frame
        joy_advanced_toggle_height_change = 50 # Approx height for one switch row
        if is_joy_model_selected:
            if not self._joy_advanced_toggle_visible:
                new_height += joy_advanced_toggle_height_change
            self.joy_advanced_toggle_frame.grid(row=current_main_frame_row, column=0, columnspan=2, sticky="ew", pady=self.DEFAULT_PADY)
            self._joy_advanced_toggle_visible = True
            current_main_frame_row += 1
        else:
            if self._joy_advanced_toggle_visible:
                new_height -= joy_advanced_toggle_height_change
            self.joy_advanced_toggle_frame.grid_remove()
            self._joy_advanced_toggle_visible = False
            if self.joy_advanced_mode_var.get(): # If Joy is deselected, turn off advanced mode
                self.joy_advanced_mode_var.set(False)


        # 6) JoyCaption Advanced Options Frame
        joy_advanced_options_height_change = 280 # Approx height for 6 entries + explanation
        show_advanced_options = is_joy_model_selected and self.joy_advanced_mode_var.get()
        if show_advanced_options:
            if not self._joy_advanced_options_visible:
                new_height += joy_advanced_options_height_change
            self.joy_advanced_options_frame.grid(row=current_main_frame_row, column=0, columnspan=2, sticky="ew", pady=self.DEFAULT_PADY)
            self._joy_advanced_options_visible = True
            current_main_frame_row += 1
        else:
            if self._joy_advanced_options_visible:
                new_height -= joy_advanced_options_height_change
            self.joy_advanced_options_frame.grid_remove()
            self._joy_advanced_options_visible = False


        # 7) Re-grid subsequent frames (Progress and Buttons)
        self.progress_frame.grid(row=current_main_frame_row, column=0, columnspan=2, sticky="ew", pady=self.DEFAULT_PADY)
        self.buttons_frame.grid(row=current_main_frame_row + 1, column=0, columnspan=2, sticky="ew", pady=self.DEFAULT_PADY)

        # Resize window if dimensions changed
        final_target_width = max(new_width, self.APP_MIN_WIDTH)
        final_target_height = max(new_height, self.APP_MIN_HEIGHT)

        if final_target_width != current_width or final_target_height != current_height:
            self.geometry(f"{final_target_width}x{final_target_height}{current_pos_str}")


        # Always keep threshold states consistent
        self._update_threshold_states()

    def _update_threshold_states(self) -> None:
        if self.general_mcut_var.get():
            self.general_threshold_var.set("0.0")
            self.general_threshold_entry.configure(state="disabled", placeholder_text="Auto")
            self.general_threshold_label.configure(text_color=("gray", "gray"))
        else:
            model = self.config_state["model_var"].get()
            self.general_threshold_var.set("0.5" if self._is_joycaption_model(model) else "0.35")
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

    # ─── Helper to check window existence ─────────────────────────────────────
    def _window_exists(self) -> bool:
        try:
            return self.winfo_exists() == 1
        except TclError:
            return False

    def _safely_update_widget(self, widget: Any, **kwargs: Any) -> None:
        try:
            if self._window_exists():
                widget.configure(**kwargs)
                widget.update()
        except (TclError, RuntimeError):
            pass

    # ─── Called by the background thread to update progress ──────────────────
    def set_progress(self, value: int, max_value: int) -> None:
        progress = value / max_value if max_value > 0 else 0
        try:
            if self._window_exists():
                self.after(0, lambda: self._update_progress_ui(progress, value, max_value))
        except (TclError, RuntimeError):
            pass

    def _update_progress_ui(self, progress: float, value: int, max_value: int) -> None:
        try:
            if self._window_exists():
                self.progress.set(progress)
                self.progress_label.configure(text=f"{value}/{max_value}")
                self.progress.update()
        except (TclError, RuntimeError):
            pass

    # ─── Main entrypoint when user clicks “Create Captions” ──────────────────
    def create_captions(self) -> None:
        """Validate inputs, load the chosen model, then run caption_folder() in a thread."""
        model_name = self.config_state["model_var"].get()
        if not model_name:
            self._show_error("No model selected", "Please select a captioning model.")
            return

        path = self.path_entry.get().strip()
        if not path or not os.path.exists(path) or not os.path.isdir(path):
            self._show_error("Invalid path", "The specified folder does not exist.")
            return

        # Parameters for model initialization (e.g., passed to __init__ or load_model)
        model_init_params: dict[str, Any] = {}
        # Parameters for per-caption generation (passed to generate_caption via generation_config)
        # Initialize with a base type, will be replaced by specific type
        generation_config_data: BaseGenerationConfig | JoyCaptionGenerationConfig | MoondreamGenerationConfig | WDGenerationConfig | BlipGenerationConfig = {}
        # The actual text prompt to be passed to the model's generate_caption method
        prompt_text_for_model: str = ""

        ui_prefix_value = self.prefix_entry.get()
        ui_postfix_value = self.postfix_entry.get()

        if self._is_wd_model(model_name):
            try:
                general_threshold_str = self.general_threshold_var.get()
                if not self.general_mcut_var.get() and not self._validate_float_range(general_threshold_str, 0, 1):
                    self._show_error("Invalid general threshold", "General threshold must be a number between 0 and 1.")
                    return
                min_general = 0.0 if self.general_mcut_var.get() else float(general_threshold_str)

                character_threshold_str = self.character_threshold_var.get()
                if not self.character_mcut_var.get() and not self._validate_float_range(character_threshold_str, 0, 1):
                    self._show_error("Invalid character threshold", "Character threshold must be a number between 0 and 1.")
                    return
                min_character = 0.15 if self.character_mcut_var.get() else float(character_threshold_str)

                model_init_params = { # These are for WD model __init__
                    "use_mcut_general": self.general_mcut_var.get(),
                    "use_mcut_character": self.character_mcut_var.get(),
                    "min_general_threshold": min_general,
                    "min_character_threshold": min_character,
                }
            except ValueError as e:
                self._show_error("Invalid threshold", f"Could not parse threshold value: {e}")
                return
            prompt_text_for_model = "" # WD models typically don't use a string prompt for generate_caption
            generation_config_data = WDGenerationConfig() # Empty config for WD

        elif self._is_moondream_model(model_name):
            caption_length = self.config_state["caption_length_var"].get()
            if caption_length not in self.config_state["caption_lengths"]:
                self._show_error("Invalid caption length", f"Caption length must be one of: {', '.join(self.config_state['caption_lengths'])}")
                return
            model_init_params = {"stream": False} # Example init param
            generation_config_data = MoondreamGenerationConfig(caption_length=caption_length)
            prompt_text_for_model = "" # Moondream handles internally based on caption_length

        elif self._is_joycaption_model(model_name):
            # Prefix and suffix are ignored for JoyCaption
            ui_prefix_value = ""
            ui_postfix_value = ""

            # Prompt is now taken directly from the live textbox
            if not hasattr(self, 'joy_live_prompt_textbox'):
                self._show_error("UI Error", "JoyCaption prompt textbox not found.")
                return
            prompt_text_for_model = self.joy_live_prompt_textbox.get("1.0", "end-1c").strip()
            if not prompt_text_for_model:
                self._show_error("JoyCaption Error", "Prompt cannot be empty for JoyCaption.")
                return

            # Advanced JoyCaption Parameters
            adv_temp_val = self.JOY_DEFAULT_TEMPERATURE
            adv_top_p_val = self.JOY_DEFAULT_TOP_P
            adv_top_k_val = self.JOY_DEFAULT_TOP_K
            adv_max_toks = JOY_DEFAULT_MAX_TOKENS
            adv_rep_penalty_val = self.JOY_DEFAULT_REPETITION_PENALTY
            adv_gpu_index_val = self.JOY_DEFAULT_GPU_INDEX

            if self.joy_advanced_mode_var.get():
                try:
                    adv_temp_val = float(self.joy_temp_var.get())
                    if not (0.0 <= adv_temp_val <= 1.0): # Typically 0-1, can be higher but often not useful
                        raise ValueError("Temperature must be between 0.0 and 1.0.")

                    adv_top_p_val = float(self.joy_topp_var.get())
                    if not (0.0 <= adv_top_p_val <= 1.0):
                        raise ValueError("Top-p must be between 0.0 and 1.0.")

                    adv_top_k_val = int(self.joy_topk_var.get())
                    if adv_top_k_val < 0:
                        raise ValueError("Top-K must be a non-negative integer.")

                    adv_max_toks = int(self.joy_max_tokens_var.get())
                    if adv_max_toks <= 0:
                        raise ValueError("Max tokens must be a positive integer.")

                    adv_rep_penalty_val = float(self.joy_rep_penalty_var.get())
                    if adv_rep_penalty_val < 0: # Typically >= 1.0
                        raise ValueError("Repetition Penalty must be non-negative.")

                    adv_gpu_index_val = int(self.joy_gpu_index_var.get())
                    # No specific range check for GPU index other than being an int

                except ValueError as e:
                    self._show_error("Invalid JoyCaption Advanced Parameter", str(e))
                    return

            model_init_params = {} # JoyCaption does not take 'stream' as an init param
            generation_config_data = JoyCaptionGenerationConfig(
                temperature=adv_temp_val,
                top_p=adv_top_p_val,
                max_tokens=adv_max_toks,
                top_k=adv_top_k_val,
                repetition_penalty=adv_rep_penalty_val,
                gpu_index=adv_gpu_index_val
            )

        elif self._is_blip_model(model_name):
            prompt_text_for_model = self.caption_entry.get() # BLIP uses "Initial Caption" as its prompt
            generation_config_data = BlipGenerationConfig() # Empty config for BLIP

        else: # Fallback for other/new models
            prompt_text_for_model = self.caption_entry.get() if hasattr(self, 'caption_entry') and self.caption_entry.winfo_ismapped() else ""
            model_init_params = {}
            generation_config_data = BaseGenerationConfig()


        try:
            if self._is_wd_model(model_name):
                self.parent.model_manager.captioning_model = None # Force re-instantiation
                self.parent.model_manager.current_captioning_model_name = None
                model_class = self.parent.model_manager._captioning_registry[model_name]
                captioning_model = model_class(
                    self.parent.model_manager.device,
                    self.parent.model_manager.precision,
                    model_name=model_name,
                    **model_init_params, # Pass only true init params
                )
                self.parent.model_manager.captioning_model = captioning_model
                self.parent.model_manager.current_captioning_model_name = model_name
            else:
                captioning_model = self.parent.model_manager.load_captioning_model(
                    model_name, **model_init_params # Pass only true init params
                )
        except Exception as e:
            self._show_error("Model loading error", f"Failed to load model {model_name}: {str(e)}")
            return

        if captioning_model is None:
            self._show_error("Model error", f"Failed to load model: {model_name}")
            return

        blacklist_text = self.blacklist_entry.get().strip()
        original_generate_caption = captioning_model.generate_caption

        # This wrapper's signature must match BaseImageCaptionModel.generate_caption
        def generate_caption_with_blacklist(
            sample: CaptionSample,
            prompt: str, # This is the prompt for the model
            generation_config: Any | None = None # This is the generation_config for the model
        ) -> str:
            raw_caption = original_generate_caption(
                sample,
                prompt=prompt,
                generation_config=generation_config
            )
            filtered_caption = filter_blacklisted_tags(
                raw_caption,
                blacklist_text,
                model_name,
                self.regex_enabled_var.get(),
            )
            logger.debug(f"Original model caption: {raw_caption}")
            logger.debug(f"Filtered caption: {filtered_caption}")
            return filtered_caption

        captioning_model.generate_caption = generate_caption_with_blacklist
        self._safely_update_widget(self.create_captions_button, state="disabled", text="Processing...")

        mode = self.config_state["mode_mapping"][self.config_state["mode_var"].get()]
        include_subdirectories = self.include_subdirectories_var.get()

        def caption_thread_func():
            try:
                captioning_model.caption_folder(
                    sample_dir=path,
                    prompt_text=prompt_text_for_model, # Pass the constructed prompt
                    mode=mode,
                    ui_prefix=ui_prefix_value,
                    ui_postfix=ui_postfix_value,
                    generation_config=generation_config_data, # Pass the typed config
                    progress_callback=self.set_progress,
                    error_callback=self._log_caption_error, # Add error callback if desired
                    include_subdirectories=include_subdirectories
                )
                if self._window_exists():
                    self.after(0, lambda: self._show_info("Process complete", "Caption generation completed successfully."))
                if self._window_exists():
                    self.after(0, self._refresh_parent_ui)

            except Exception as exception:
                logger.error(f"Error during caption generation: {exception}")
                error_message = str(exception)
                if self._window_exists():
                    self.after(0, lambda: self._show_error("Caption generation error", f"Error during processing: {error_message}"))
            finally:
                # Restore original generate_caption and re-enable button
                captioning_model.generate_caption = original_generate_caption
                if self._window_exists():
                    self.after(0, lambda: self._safely_update_widget(
                        self.create_captions_button, state="normal", text="Create Captions"
                    ))

        caption_thread = threading.Thread(target=caption_thread_func, daemon=True)
        caption_thread.start()

    def _refresh_parent_ui(self) -> None:
        """After captions are done, reload the image grid in the parent (if possible)."""
        try:
            if self._window_exists() and hasattr(self.parent, "image_handler") and hasattr(self.parent.image_handler, "load_image_data"):
                self.parent.image_handler.load_image_data()
                self.parent.refresh_ui()
        except (TclError, RuntimeError, AttributeError):
            pass

    def _validate_float_range(self, value: str, min_val: float, max_val: float) -> bool:
        """Return True if value is a float in [min_val, max_val]."""
        try:
            float_val = float(value)
            return min_val <= float_val <= max_val
        except ValueError:
            return False

    def _show_error(self, title: str, message: str) -> None:
        """Show an error dialog and refocus."""
        messagebox.showerror(title, message)
        try:
            if self._window_exists():
                self.focus_set()
        except (TclError, RuntimeError):
            pass

    def _log_caption_error(self, filename: str) -> None:
        logger.warning(f"Skipped captioning due to error for: {filename}")

    def _show_info(self, title: str, message: str) -> None:
        """Show an info dialog and refocus."""
        messagebox.showinfo(title, message)
        try:
            if self._window_exists():
                self.focus_set()
        except (TclError, RuntimeError):
            pass
