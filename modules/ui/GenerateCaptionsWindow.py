import logging
import os
import threading
from tkinter import TclError, filedialog, messagebox
from typing import Any, TypedDict

from modules.module.captioning.captioning_util import (
    filter_blacklisted_tags,
    get_blacklist_tags,
)
from modules.module.JoyCaptionModel import (
    CAPTION_LENGTH_CHOICES,
    CAPTION_TYPE_MAP,
    DEFAULT_MAX_TOKENS,
    JoyCaptionModel,
)
from modules.util.ui.ui_utils import set_window_icon

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

    APP_MIN_WIDTH: int = 390
    APP_MIN_HEIGHT: int = 510

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

        # Create each sub‐frame
        self.basic_options_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.caption_options_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.joycaption_options_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.threshold_frame = ctk.CTkFrame(self.main_frame)
        self.moondream_options_frame = ctk.CTkFrame(self.main_frame)
        self.additional_options_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.progress_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.buttons_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")

        # Put the first two frames in a fixed vertical order;
        # the joycaption, threshold, and moondream frames will be shown/hidden dynamically
        frames = [
            self.basic_options_frame,
            self.caption_options_frame,
            # ‼ We do not grid the joycaption/threshold/moondream frames here; they are placed by _update_model_specific_options()
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

        # Now build each section
        self._create_basic_options(path)
        self._create_caption_configuration()
        self._create_joycaption_configuration()         # ← New
        self._create_threshold_configuration()
        self._create_moondream_configuration()
        self._create_additional_options(include_subdirectories)
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
        self.prefix_entry = ctk.CTkEntry(self.caption_options_frame)
        self._create_labeled_widget(self.caption_options_frame, "Caption Prefix", self.prefix_entry, row=1)

        # Row 2: Caption Suffix
        self.postfix_entry = ctk.CTkEntry(self.caption_options_frame)
        self._create_labeled_widget(self.caption_options_frame, "Caption Suffix", self.postfix_entry, row=2)

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
         - Extra Options   (comma-separated entry)
         - Name (for NAME_OPTION)
         - Temperature, Top-p, Max Tokens
        """
        row = 0
        # Caption Type dropdown
        self.joy_caption_type_var = ctk.StringVar(self, list(CAPTION_TYPE_MAP.keys())[0])
        self.joy_caption_type_dropdown = ctk.CTkOptionMenu(
            self.joycaption_options_frame,
            variable=self.joy_caption_type_var,
            values=list(CAPTION_TYPE_MAP.keys()),
            dynamic_resizing=False,
            width=200,
        )
        self._create_labeled_widget(
            self.joycaption_options_frame, "Caption Type", self.joy_caption_type_dropdown, row=row
        )

        # Caption Length dropdown
        row += 1
        self.joy_caption_length_var = ctk.StringVar(self, CAPTION_LENGTH_CHOICES[0])
        self.joy_caption_length_dropdown = ctk.CTkOptionMenu(
            self.joycaption_options_frame,
            variable=self.joy_caption_length_var,
            values=CAPTION_LENGTH_CHOICES,
            dynamic_resizing=False,
            width=120,
        )
        self._create_labeled_widget(
            self.joycaption_options_frame, "Caption Length", self.joy_caption_length_dropdown, row=row
        )

        # Extra Options (comma-separated)
        row += 1
        self.joy_extra_options_entry = ctk.CTkEntry(self.joycaption_options_frame)
        self._create_labeled_widget(
            self.joycaption_options_frame,
            "Extra Options (comma-sep.)",
            self.joy_extra_options_entry,
            row=row,
        )
        help_txt = "Type any EXTRA_OPTIONS (from JoyCaptionModel) as comma-separated fragments."
        self._create_explanation_label(self.joycaption_options_frame, help_txt, row=row+1)

        # Name input
        row += 2
        self.joy_name_entry = ctk.CTkEntry(self.joycaption_options_frame)
        self._create_labeled_widget(
            self.joycaption_options_frame, "Name (for NAME_OPTION)", self.joy_name_entry, row=row
        )

        # Temperature
        row += 1
        self.joy_temperature_var = ctk.StringVar(self, str(JoyCaptionModel.DEFAULT_TEMPERATURE) if hasattr(JoyCaptionModel, "DEFAULT_TEMPERATURE") else "0.6")
        self.joy_temperature_entry = ctk.CTkEntry(self.joycaption_options_frame, textvariable=self.joy_temperature_var)
        self._create_labeled_widget(
            self.joycaption_options_frame, "Temperature (0.0–1.0)", self.joy_temperature_entry, row=row
        )

        # Top‐p
        row += 1
        self.joy_topp_var = ctk.StringVar(self, str(JoyCaptionModel.DEFAULT_TOP_P) if hasattr(JoyCaptionModel, "DEFAULT_TOP_P") else "0.9")
        self.joy_topp_entry = ctk.CTkEntry(self.joycaption_options_frame, textvariable=self.joy_topp_var)
        self._create_labeled_widget(
            self.joycaption_options_frame, "Top-p (0.0–1.0)", self.joy_topp_entry, row=row
        )

        # Max Tokens
        row += 1
        self.joy_max_tokens_var = ctk.StringVar(self, str(DEFAULT_MAX_TOKENS))
        self.joy_max_tokens_entry = ctk.CTkEntry(self.joycaption_options_frame, textvariable=self.joy_max_tokens_var)
        self._create_labeled_widget(
            self.joycaption_options_frame, "Max Tokens", self.joy_max_tokens_entry, row=row
        )

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
            self.buttons_frame, text="Create Captions", command=self.create_captions
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

        # Update Initial Caption field state
        if hasattr(self, 'initial_caption_label') and hasattr(self, 'caption_entry'):
            if self._is_blip_model(model):
                # Show and enable for Blip2 models
                self.initial_caption_label.grid(row=0, column=0, **self.LABEL_GRID)
                self.caption_entry.grid(row=0, column=1, **self.WIDGET_GRID)
                try:
                    # Attempt to get the default label color from the theme
                    default_text_color = ctk.ThemeManager.theme["CTkLabel"]["text_color"]
                except (KeyError, AttributeError):  # Fallback if theme not fully loaded or key missing
                    default_text_color = ("#000000", "#FFFFFF") # Black for light, White for dark
                self.initial_caption_label.configure(text_color=default_text_color)
                self.caption_entry.configure(state="short", placeholder_text="")
            else:
                # Hide for all other models
                self.initial_caption_label.grid_remove()
                self.caption_entry.grid_remove()
                self.caption_entry.delete(0, "end") # Clear content when hidden

        # Get current window dimensions for resizing
        geometry = self.geometry().split("+")[0]
        current_width, current_height = map(int, geometry.split("x"))
        new_height = current_height # Start with current height, adjust based on frame visibility changes

        # --- Dynamic frames positioning ---
        # Base row index in main_frame for the first dynamic/optional frame
        next_dynamic_row_index = 2

        # 1) WD threshold frame
        is_wd = self._is_wd_model(model)
        threshold_height_change = 110  # Approximate height for this frame
        if is_wd:
            if not self._threshold_visible:  # Frame is becoming visible
                new_height += threshold_height_change
            self.threshold_frame.grid(row=next_dynamic_row_index, column=0, columnspan=2, sticky="ew", pady=self.DEFAULT_PADY)
            self._threshold_visible = True
            next_dynamic_row_index += 1
        else:  # Frame should be hidden
            if self._threshold_visible:  # Frame is becoming hidden
                new_height -= threshold_height_change
            self.threshold_frame.grid_remove()
            self._threshold_visible = False

        # 2) Moondream frame (legacy)
        is_moondream = self._is_moondream_model(model)
        moondream_height_change = 70  # Approximate height (was 110)
        if is_moondream:
            if not self._moondream_options_visible:  # Frame is becoming visible
                new_height += moondream_height_change
            self.moondream_options_frame.grid(row=next_dynamic_row_index, column=0, columnspan=2, sticky="ew", pady=self.DEFAULT_PADY)
            self._moondream_options_visible = True
            next_dynamic_row_index += 1
        else:  # Frame should be hidden
            if self._moondream_options_visible:  # Frame is becoming hidden
                new_height -= moondream_height_change
            self.moondream_options_frame.grid_remove()
            self._moondream_options_visible = False

        # 3) JoyCaption frame
        is_joy = self._is_joycaption_model(model)
        joy_height_change = 280  # Approximate height (was 240)
        if is_joy:
            if not self._joycaption_visible:  # Frame is becoming visible
                new_height += joy_height_change
            self.joycaption_options_frame.grid(row=next_dynamic_row_index, column=0, columnspan=2, sticky="ew", pady=self.DEFAULT_PADY)
            self._joycaption_visible = True
            next_dynamic_row_index += 1
        else:  # Frame should be hidden
            if self._joycaption_visible:  # Frame is becoming hidden
                new_height -= joy_height_change
            self.joycaption_options_frame.grid_remove()
            self._joycaption_visible = False

        # 4) Re-grid subsequent frames based on the final next_dynamic_row_index
        self.additional_options_frame.grid(row=next_dynamic_row_index, column=0, columnspan=2, sticky="ew", pady=self.DEFAULT_PADY)
        self.progress_frame.grid(row=next_dynamic_row_index + 1, column=0, columnspan=2, sticky="ew", pady=self.DEFAULT_PADY)
        self.buttons_frame.grid(row=next_dynamic_row_index + 2, column=0, columnspan=2, sticky="ew", pady=self.DEFAULT_PADY)

        # Resize window if height changed
        if new_height != current_height:
            # Use the class constant APP_MIN_HEIGHT directly instead of self.minsize()[1]
            self.geometry(f"{current_width}x{max(new_height, self.APP_MIN_HEIGHT)}")

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
        # 1) Validate model selection
        model_name = self.config_state["model_var"].get()
        if not model_name:
            self._show_error("No model selected", "Please select a captioning model.")
            return

        # 2) Validate path
        path = self.path_entry.get().strip()
        if not path or not os.path.exists(path) or not os.path.isdir(path):
            self._show_error("Invalid path", "The specified folder does not exist.")
            return

        # 3) Decide per‐model model_kwargs
        model_kwargs: dict[str, Any] = {}

        if self._is_wd_model(model_name):
            # WD‐style thresholds
            try:
                general_threshold_str = self.general_threshold_var.get()
                if (
                    not self.general_mcut_var.get()
                    and not self._validate_float_range(general_threshold_str, 0, 1)
                ):
                    self._show_error(
                        "Invalid general threshold",
                        "General threshold must be a number between 0 and 1.",
                    )
                    return
                min_general = 0.0 if self.general_mcut_var.get() else float(general_threshold_str)

                character_threshold_str = self.character_threshold_var.get()
                if (
                    not self.character_mcut_var.get()
                    and not self._validate_float_range(character_threshold_str, 0, 1)
                ):
                    self._show_error(
                        "Invalid character threshold",
                        "Character threshold must be a number between 0 and 1.",
                    )
                    return
                min_character = (
                    0.15 if self.character_mcut_var.get() else float(character_threshold_str)
                )

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
            # Moondream2: only a “caption_length” (short/normal)
            caption_length = self.config_state["caption_length_var"].get()
            if caption_length not in self.config_state["caption_lengths"]:
                self._show_error(
                    "Invalid caption length",
                    f"Caption length must be one of: {', '.join(self.config_state['caption_lengths'])}",
                )
                return

            model_kwargs = {
                "caption_length": caption_length,
                "stream": True,
            }

        elif self._is_joycaption_model(model_name):
            # JoyCaptionModel: read all new UI fields
            #  a) caption_type
            caption_type = self.joy_caption_type_var.get()
            if caption_type not in CAPTION_TYPE_MAP:
                self._show_error(
                    "Invalid caption type",
                    f"Caption type must be one of: {', '.join(CAPTION_TYPE_MAP.keys())}",
                )
                return

            #  b) caption_length
            caption_length = self.joy_caption_length_var.get()
            if caption_length not in CAPTION_LENGTH_CHOICES:
                self._show_error(
                    "Invalid caption length",
                    f"Caption length must be one of: {', '.join(CAPTION_LENGTH_CHOICES)}",
                )
                return

            #  c) extra_options (parse comma-separated)
            raw_extra = self.joy_extra_options_entry.get().strip()
            extra_options = [opt.strip() for opt in raw_extra.split(",") if opt.strip()] if raw_extra else []

            #  d) name_input
            name_input = self.joy_name_entry.get().strip()

            #  e) temperature
            try:
                temp_val = float(self.joy_temperature_var.get())
                if not (0.0 <= temp_val <= 1.0):
                    raise ValueError("Temperature out of range")
            except Exception:
                self._show_error(
                    "Invalid temperature",
                    "Temperature must be a float between 0.0 and 1.0.",
                )
                return

            #  f) top-p
            try:
                top_p_val = float(self.joy_topp_var.get())
                if not (0.0 <= top_p_val <= 1.0):
                    raise ValueError("Top-p out of range")
            except Exception:
                self._show_error(
                    "Invalid top-p",
                    "Top-p must be a float between 0.0 and 1.0.",
                )
                return

            #  g) max_tokens
            try:
                max_toks = int(self.joy_max_tokens_var.get())
                if max_toks <= 0:
                    raise ValueError("Max tokens must be positive")
            except Exception:
                self._show_error(
                    "Invalid max tokens",
                    "Max Tokens must be a positive integer.",
                )
                return

            model_kwargs = {
                "caption_type": caption_type,
                "caption_length": caption_length,
                "extra_options": extra_options,
                "name_input": name_input,
                "temperature": temp_val,
                "top_p": top_p_val,
                "max_tokens": max_toks,
                "stream": True,
            }

        # 4) Load or initialize the appropriate model
        try:
            if self._is_wd_model(model_name):
                # WD models are instantiated directly from the registry
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
                # For Moondream, JoyCaption, or any other “loadable” model
                captioning_model = self.parent.model_manager.load_captioning_model(
                    model_name, **model_kwargs
                )
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            self._show_error("Model loading error", f"Failed to load model {model_name}: {str(e)}")
            return

        if captioning_model is None:
            logger.error(f"Failed to load model: {model_name}")
            self._show_error("Model error", f"Failed to load model: {model_name}")
            return

        # 5) Prepare blacklist (only once)
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

        # 6) Wrap the model's generate_caption to apply blacklist filtering
        original_generate_caption = captioning_model.generate_caption

        def generate_caption_with_blacklist(
            sample,                 # Receives actual caption_sample object
            initial: str = "",      # Receives actual initial_caption string from the caller
            initial_caption: str = "", # This wrapper param receives actual caption_prefix from the caller
            caption_prefix: str = "",  # This wrapper param receives actual caption_postfix from the caller
            caption_postfix: str = "", # This wrapper param defaults to "" and is unused if caller provides 4 args
            **kwargs                # All other model-specific args (empty in this call path)
        ):
            # Call original_generate_caption with its expected named arguments.
            # `original_generate_caption` is JoyCaptionModel.generate_caption, which expects:
            # (self, caption_sample, initial_caption, caption_prefix, caption_postfix)
            caption = original_generate_caption(
                sample,                                  # This is the actual caption_sample object
                initial_caption=initial,                 # `initial` holds the actual initial_caption string
                caption_prefix=initial_caption,          # wrapper's `initial_caption` param holds the actual caption_prefix string
                caption_postfix=caption_prefix           # wrapper's `caption_prefix` param holds the actual caption_postfix string
                                                         # **kwargs is omitted as original_generate_caption doesn't accept it
            )
            filtered_caption = filter_blacklisted_tags(
                caption,
                blacklist_text, # This is from the outer scope of create_captions
                model_name,     # This is from the outer scope
                self.regex_enabled_var.get(), # Outer scope
            )
            logger.debug(f"Original caption: {caption}")
            logger.debug(f"Filtered caption: {filtered_caption}")
            return filtered_caption

        captioning_model.generate_caption = generate_caption_with_blacklist

        # 7) Disable the button while processing
        self._safely_update_widget(self.create_captions_button, state="disabled", text="Processing...")

        # 8) Gather common parameters
        mode = self.config_state["mode_mapping"][self.config_state["mode_var"].get()]
        initial_caption = self.caption_entry.get()
        caption_prefix = self.prefix_entry.get()
        caption_postfix = self.postfix_entry.get()
        include_subdirectories = self.include_subdirectories_var.get()

        # 9) Run caption_folder() on a background thread
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
                    **model_kwargs # Pass the collected model_kwargs here
                )
                # On success, show info and refresh parent UI
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

    def _show_info(self, title: str, message: str) -> None:
        """Show an info dialog and refocus."""
        messagebox.showinfo(title, message)
        try:
            if self._window_exists():
                self.focus_set()
        except (TclError, RuntimeError):
            pass
