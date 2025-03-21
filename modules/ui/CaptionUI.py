"""
OneTrainer Dataset Tools

This module defines a CTkToplevel window for image editing with the capbility to automatically
or manually caption and mask images from a loaded directory.
"""

import contextlib
import logging
import platform
import re
import subprocess
import tkinter as tk
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from tkinter import filedialog
from typing import TypeAlias

from modules.module.Blip2Model import Blip2Model
from modules.module.BooruModels import BooruModels
from modules.module.ClipSegModel import ClipSegModel
from modules.module.MaskByColor import MaskByColor
from modules.module.Moondream2Model import Moondream2Model
from modules.module.RembgHumanModel import RembgHumanModel
from modules.module.RembgModel import RembgModel
from modules.module.SAMoondreamModel import MoondreamSAMMaskModel
from modules.ui.BulkCaptionEditWindow import BulkCaptionEditWindow
from modules.ui.FileOperationsWindow import FileOperationsWindow
from modules.ui.GenerateCaptionsWindow import GenerateCaptionsWindow
from modules.ui.GenerateMasksWindow import GenerateMasksWindow
from modules.util import path_util
from modules.util.torch_util import default_device
from modules.util.ui import components
from modules.util.ui.icons import load_icon
from modules.util.ui.ui_utils import bind_mousewheel
from modules.util.ui.UIState import UIState

import torch

import customtkinter as ctk
import cv2
import numpy as np
import pillow_jxl  # noqa: F401  # Needed for plugin registration
from customtkinter import ThemeManager
from PIL import Image, ImageDraw, ImageTk

logger = logging.getLogger(__name__)


MAX_VISIBLE_FILES: int = 50
IMAGE_PADDING: int = 20
DEFAULT_IMAGE_WIDTH: int = 1600
DEFAULT_IMAGE_HEIGHT: int = 1200

MIN_BRUSH_SIZE: float = 0.0025
MAX_BRUSH_SIZE: float = 0.5
BRUSH_SIZE_CHANGE_FACTOR: float = 1.25
MASK_HISTORY_LIMIT: int = 30
BRUSH_SIZE_WHEEL_SMALL_FACTOR: float = 0.03
BRUSH_SIZE_WHEEL_LARGE_FACTOR: float = 0.05
BRUSH_SIZE_THRESHOLD: float = 0.05
MIN_CONTAINER_SIZE: int = 10
BRACKET_BRUSH_SIZE_STEP: float = 0.01

WINDOW_WIDTH: int = 1184
WINDOW_HEIGHT: int = 704
FILE_LIST_WIDTH: int = 300
MASK_MIN_OPACITY: float = 0.3
DEFAULT_BRUSH_SIZE: float = 0.03


THEME_COLORS = {
    "dark": {
        "canvas_bg": (32, 32, 32),
        "main_frame_bg": "#242424",
        "image_container_bg": "#242424",
        "tools_frame_bg": "#333333",
        "caption_frame_bg": "#333333",
        "selected_file_bg": "#454545",
        "selected_file_text": "#5AD9ED",
        "transparent": "transparent",
        "file_text": "white",
    },
    "light": {
        "canvas_bg": "#EBEBEB",
        "main_frame_bg": "#EBEBEB",
        "image_container_bg": "#DBDBDB",
        "tools_frame_bg": "#cccccc",
        "caption_frame_bg": "#cccccc",
        "selected_file_bg": "#bababa",
        "selected_file_text": "#0056aa",
        "transparent": "transparent",
        "file_text": "#2d2d2d",
    }
}

CANVAS_BACKGROUND_COLOR: tuple[int, int, int] = THEME_COLORS["dark"]["canvas_bg"]


RGBColor: TypeAlias = tuple[int, int, int]
ImageCoordinates: TypeAlias = tuple[int, int, int, int]

def get_theme_color(color_name: str) -> str | tuple[int, int, int]:
    """Get the appropriate color based on current theme."""
    appearance_mode = ctk.get_appearance_mode().lower()
    theme = "dark" if appearance_mode == "dark" else "light"
    return THEME_COLORS[theme][color_name]

def get_themed_icon(icon_name: str, size: tuple[int, int]) -> ImageTk.PhotoImage:
    """Get an icon with theme-specific variant if available."""
    appearance_mode = ctk.get_appearance_mode().lower()
    if appearance_mode == "light" and icon_name == "folder-tree":
        # Try loading the dark variant for better contrast in light mode
        try:
            return load_icon(f"{icon_name}_dark", size)
        except (FileNotFoundError, OSError, ValueError):
            pass
    return load_icon(icon_name, size)

def natural_sort_key(s):
    """Sort strings with embedded numbers in natural order."""
    # Split the input string into text and numeric parts
    def convert(text):
        return int(text) if text.isdigit() else text.lower()
    return [convert(c) for c in re.split(r'(\d+)', s)]

def scan_for_supported_images(
    directory: Path,
    include_subdirs: bool,
    is_supported: Callable[[Path], bool],
) -> list[Path]:
    directory = Path(directory)
    if include_subdirs:
        results = [
            p.relative_to(directory)
            for p in directory.glob('**/*')
            if p.is_file() and is_supported(p)
        ]
    else:
        results = [
            p.name
            for p in directory.iterdir()
            if p.is_file() and is_supported(p)
        ]
    # Use natural sorting instead of lexicographical sorting
    return sorted(results, key=natural_sort_key)

def get_platform_cursor(cursor_name: str, fallback_cursor: str) -> str:
    """Get platform-specific cursor representation."""
    # Use global cursor cache
    if not hasattr(get_platform_cursor, "cursor_cache"):
        get_platform_cursor.cursor_cache = {}

    # Return from cache if available
    cache_key = f"{cursor_name}:{fallback_cursor}"
    if cache_key in get_platform_cursor.cursor_cache:
        return get_platform_cursor.cursor_cache[cache_key]

    # Standard cursor names across platforms
    standard_cursors = {
        "brush": "pencil",
        "fill": "dotbox",
    }

    # First check for custom cursor files on Windows
    if platform.system() == "Windows":
        cursor_path = Path(__file__).parent.parent.parent / "resources" / "icons" / "cursors" / f"cursor_{cursor_name}.cur"
        if cursor_path.exists():
            # Use forward slashes instead of backslashes for Tk or it will freak out
            normalized_path = str(cursor_path).replace("\\", "/")
            result = f"@{normalized_path}"
        else:
            # Fall back to standard cursor mapping
            result = standard_cursors.get(cursor_name, fallback_cursor)
    else:
        # Use standard cursor mapping for non-Windows platforms
        result = standard_cursors.get(cursor_name, fallback_cursor)

    # Cache the result
    get_platform_cursor.cursor_cache[cache_key] = result
    return result


class CaptionUI(ctk.CTkToplevel):


    def __init__(
        self,
        parent: ctk.CTk,
        initial_dir: str | None = None,
        include_subdirectories: bool = False,
        *args,
        **kwargs,
    ) -> None:
        logger.debug("CaptionUI __init__ started. Parent: %s", parent)
        super().__init__(parent, *args, **kwargs)
        self.attributes("-topmost", True)

        # Set up explicit parent reference for proper stacking behavior
        self.parent = parent
        self.dir: str | None = initial_dir
        self.config_ui_data: dict = {"include_subdirectories": include_subdirectories}
        self.config_ui_state: UIState = UIState(self, self.config_ui_data)

        # Initialize other attributes and managers
        self.image_rel_paths: list[str] = []
        self.current_image_index: int = -1
        self.pil_image: Image.Image | None = None
        self.pil_mask: Image.Image | None = None
        self.image_width: int = 0
        self.image_height: int = 0
        self.caption_lines: list[str] = []
        self.current_caption_line: int = 0

        self.masking_model = None
        self.captioning_model = None

        self.mask_editor = MaskEditor(self)
        self.model_manager = ModelManager()
        self.navigation_manager = NavigationManager(self)
        self.caption_manager = CaptionManager(self)
        self.file_manager = FileManager(self)
        self.image_handler = ImageHandler(self)

        self._setup_window()

        self._create_layout()

        if initial_dir:
            logger.debug("Loading initial directory: %s", initial_dir)
            self.file_manager.load_directory()

    def _setup_window(self) -> None:
        """Configure the window's basic properties."""
        self.title("Dataset Tools")
        self.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.resizable(True, True)
        self.protocol("WM_DELETE_WINDOW", self.destroy)

        # After a short delay, remove the topmost attribute so the window behaves normally.
        self.after(200, lambda: self.attributes("-topmost", False))

        self.help_text: str = (
                "Keyboard shortcuts:\n\n"
                "⬅️➡️: Navigate between images\n"
                "Tab: Switch between caption lines\n"
                "Return or Ctrl+S: Save changes\n"
                "Ctrl+M: Toggle mask display\n"
                "Ctrl+D: Switch to draw mode\n"
                "Ctrl+F: Switch to fill mode\n"
                "Ctrl+Z: Undo mask edit\n"
                "Ctrl+Y: Redo mask edit\n"
                "[ or ]: Decrease/increase brush size by 0.1\n\n"
                "When editing masks:\n"
                "Left click: Add to mask\n"
                "Right click: Remove from mask\n"
                "Mouse wheel: Adjust brush size"
                "Middle mouse click: Reset image (clear caption and mask, save to commit change)\n"
            )

    def _create_layout(self) -> None:
        """Create the main UI layout."""
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.enable_mask_editing_var = ctk.BooleanVar(value=False)
        self._create_top_bar()
        self._create_main_content()

    def _create_icon_button(
        self,
        parent: ctk.CTkFrame,
        row: int,
        column: int,
        text: str,
        command,
        icon_name: str,
        tooltip: str,
    ) -> None:
        """Helper to create an icon button."""
        icon = load_icon(icon_name, (24, 24))
        components.icon_button(
            parent, row, column, text, command, image=icon, tooltip=tooltip
        )

    def _create_top_bar(self) -> None:
        """Create the top toolbar."""
        top_frame = ctk.CTkFrame(self, corner_radius=0)
        top_frame.grid(row=0, column=0, sticky="new")
        self._create_icon_button(
            top_frame,
            0,
            0,
            "Load",
            self.file_manager.open_directory,
            "load",
            "Load a directory of images",
        )
        self._create_icon_button(
            top_frame,
            0,
            1,
            "Auto-Mask",
            self._open_mask_window,
            "auto-mask",
            "Generate masks automatically",
        )
        self._create_icon_button(
            top_frame,
            0,
            2,
            "Auto-Caption",
            self._open_caption_window,
            "auto-caption",
            "Generate captions automatically",
        )
        self._create_icon_button(
            top_frame,
            0,
            3,
            "Browse",
            self.file_manager.open_in_explorer,
            "explorer",
            "Open in File Browser",
        )
        components.switch(
            top_frame,
            0,
            4,
            self.config_ui_state,
            "include_subdirectories",
            text="Include Subdirs",
            tooltip="Include subdirectories when loading images",
        )
        self._create_icon_button(
            top_frame,
            0,
            5,
            "Image Tools",
            self._open_file_tools,
            "file-cog",
            "Open file operations tools",
        )
        self._create_icon_button(
            top_frame,
            0,
            6,
            "Bulk Caption Edit",
            self._open_bulk_caption_edit,
            "bulk",
            "Apply bulk transformations to captions at once"
        )
        top_frame.grid_columnconfigure(7, weight=1)
        self._create_icon_button(
            top_frame,
            0,
            8,
            "Help",
            self._show_help,
            "help",
            self.help_text,
        )

    def _create_main_content(self) -> None:
        """Create the main content area."""
        main_frame = ctk.CTkFrame(self, fg_color=get_theme_color("main_frame_bg"))
        main_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=10)
        main_frame.grid_columnconfigure(0, weight=0)
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)
        self._create_file_list(main_frame)
        self._create_editor_panel(main_frame)

    def _create_file_list(self, parent: ctk.CTkFrame) -> None:
        """Create the file list pane with fixed headers."""
        # Main container for file area
        file_area_frame = ctk.CTkFrame(parent, fg_color="transparent")
        file_area_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        file_area_frame.grid_columnconfigure(0, weight=1)

        # Configure rows: fixed headers (0-2) and scrollable list (3)
        file_area_frame.grid_rowconfigure(0, weight=0)  # Header
        file_area_frame.grid_rowconfigure(1, weight=0)  # File filter
        file_area_frame.grid_rowconfigure(2, weight=0)  # Caption filter
        file_area_frame.grid_rowconfigure(3, weight=1)  # Scrollable file list

        # 1. Create header frame (folder display)
        header_frame = ctk.CTkFrame(file_area_frame)
        header_frame.grid(row=0, column=0, sticky="ew", padx=2, pady=(2, 4))
        header_frame.grid_columnconfigure(1, weight=1)
        folder_icon = get_themed_icon("folder-tree", (20, 20))
        ctk.CTkLabel(header_frame, text="", image=folder_icon).grid(
            row=0, column=0, padx=(5, 3), pady=5
        )
        self.folder_name_label = ctk.CTkLabel(
            header_frame,
            text="No folder selected",
            anchor="w",
            font=("Segoe UI", 12, "bold"),
            wraplength=FILE_LIST_WIDTH - 35,
        )
        self.folder_name_label.grid(
            row=0, column=1, sticky="ew", padx=0, pady=5
        )

        # Initialize debounce timer attribute
        self.filter_debounce_timer = None
        self.FILTER_DEBOUNCE_DELAY = 300  # milliseconds

        # 2. Create file/path filter frame
        file_filter_frame = ctk.CTkFrame(file_area_frame)
        file_filter_frame.grid(row=1, column=0, sticky="ew", padx=2, pady=(2, 4))
        file_filter_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(file_filter_frame, text="File/Path Filter:", anchor="w").grid(
            row=0, column=0, columnspan=2, sticky="w", padx=5, pady=(5, 0)
        )

        self.file_filter_var = ctk.StringVar(value="")
        self.file_filter_entry = ctk.CTkEntry(file_filter_frame, textvariable=self.file_filter_var)
        self.file_filter_entry.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        self.file_filter_entry.bind("<KeyRelease>", lambda e: self._debounce_filter())

        self.file_filter_type_var = ctk.StringVar(value="File")
        self.file_filter_type = ctk.CTkOptionMenu(
            file_filter_frame,
            values=["File", "Path", "Both"],
            variable=self.file_filter_type_var,
            width=80,
            command=lambda _: self._apply_filters()  # Dropdown changes apply immediately
        )
        self.file_filter_type.grid(row=1, column=1, sticky="e", padx=(3, 5), pady=5)

        # 3. Create caption filter frame
        caption_filter_frame = ctk.CTkFrame(file_area_frame)
        caption_filter_frame.grid(row=2, column=0, sticky="ew", padx=2, pady=(2, 4))
        caption_filter_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(caption_filter_frame, text="Caption Filter:", anchor="w").grid(
            row=0, column=0, columnspan=2, sticky="w", padx=5, pady=(5, 0)
        )

        self.caption_filter_var = ctk.StringVar(value="")
        self.caption_filter_entry = ctk.CTkEntry(caption_filter_frame, textvariable=self.caption_filter_var)
        self.caption_filter_entry.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        self.caption_filter_entry.bind("<KeyRelease>", lambda e: self._debounce_filter())

        self.caption_filter_type_var = ctk.StringVar(value="Contains")
        self.caption_filter_type = ctk.CTkOptionMenu(
            caption_filter_frame,
            values=["Contains", "Matches", "Excludes", "Regex"],
            variable=self.caption_filter_type_var,
            width=80,
            command=lambda _: self._apply_filters()  # Dropdown changes apply immediately
        )
        self.caption_filter_type.grid(row=1, column=1, sticky="e", padx=(3, 5), pady=5)

        # 4. Create scrollable file list
        self.file_list = ctk.CTkScrollableFrame(
            file_area_frame,
            width=FILE_LIST_WIDTH,
            scrollbar_fg_color="transparent",
            scrollbar_button_hover_color="grey75",
            scrollbar_button_color="grey50",
        )
        self.file_list.grid(row=3, column=0, sticky="nsew")
        self._file_list_scrollbar = self.file_list._scrollbar

        # Initialize file list variables
        self.image_labels = []
        self.filtered_image_paths = []

    def _debounce_filter(self) -> None:
        """Debounce the filter application to avoid lag during typing."""
        # Cancel any pending timer
        if hasattr(self, 'filter_debounce_timer') and self.filter_debounce_timer:
            self.after_cancel(self.filter_debounce_timer)

        # Set a new timer
        self.filter_debounce_timer = self.after(
            self.FILTER_DEBOUNCE_DELAY,
            self._apply_filters
        )

    def _create_file_item(self, row_idx, file_path):
        """Create a single file item in the list."""
        safe_filename = (
            file_path if isinstance(file_path, str) else file_path.decode("utf-8", errors="replace")
        )

        # Create the label
        label = ctk.CTkLabel(
            self.file_list,
            text=safe_filename,
            wraplength=FILE_LIST_WIDTH - 20,
            font=("Segoe UI", 11),
            text_color=get_theme_color("file_text")
        )

        # Highlight if this is the current file
        is_current = self.image_rel_paths.index(file_path) == self.current_image_index if self.current_image_index >= 0 else False
        if is_current:
            label.configure(
                text_color=get_theme_color("selected_file_text"),
                fg_color=get_theme_color("selected_file_bg"),
                corner_radius=6
            )

        # Create click handler with proper closure
        def click_handler(e, file_path=file_path):
            idx = self.image_rel_paths.index(file_path)
            self.navigation_manager.switch_to_image(idx, from_click=True)

        label.bind("<Button-1>", click_handler)

        # Grid in the next row
        label.grid(
            row=row_idx,
            column=0,
            sticky="w",
            padx=2,
            pady=(1 if row_idx == 0 else 2, 2),
        )

        return label

    def _apply_filters(self) -> None:
        """Apply filters to the file list and update the display."""
        if not hasattr(self, "image_rel_paths") or not self.image_rel_paths:
            return

        self.filtered_image_paths = self._filter_files(self.image_rel_paths)
        self._update_file_list_display()

    def _filter_files(self, files: list) -> list:
        """Apply both filters to the file list."""
        # Apply file/path filter
        file_filter = self.file_filter_var.get().strip()
        filter_type = self.file_filter_type_var.get()

        filtered = files
        if file_filter:
            try:
                pattern = re.compile(re.escape(file_filter), re.IGNORECASE)

                if filter_type == "File":
                    filtered = [f for f in filtered if pattern.search(Path(f).name)]
                elif filter_type == "Path":
                    filtered = [f for f in filtered if pattern.search(str(f))]
                else:  # Both
                    filtered = [f for f in filtered if pattern.search(str(f)) or pattern.search(Path(f).name)]

            except re.error:
                # In case of regex error, don't filter
                pass

        # Apply caption filter
        caption_filter = self.caption_filter_var.get().strip()
        caption_filter_type = self.caption_filter_type_var.get()

        if caption_filter and self.dir:
            try:
                caption_files = []

                for file_path in filtered:
                    full_path = Path(self.dir) / file_path
                    caption_path = full_path.with_suffix(".txt")

                    if not caption_path.exists():
                        # Skip files without captions if we're filtering by caption
                        continue

                    try:
                        caption_content = caption_path.read_text(encoding="utf-8").strip()

                        if caption_filter_type == "Contains":
                            if caption_filter.lower() in caption_content.lower():
                                caption_files.append(file_path)
                        elif caption_filter_type == "Matches":
                            if caption_filter.lower() == caption_content.lower():
                                caption_files.append(file_path)
                        elif caption_filter_type == "Excludes":
                            if caption_filter.lower() not in caption_content.lower():
                                caption_files.append(file_path)
                        elif caption_filter_type == "Regex":
                            pattern = re.compile(caption_filter, re.IGNORECASE)
                            if pattern.search(caption_content):
                                caption_files.append(file_path)
                    except Exception:
                        # If reading fails, skip this file
                        continue

                filtered = caption_files

            except Exception as e:
                logger.error(f"Error applying caption filter: {e}")

        return filtered

    def _update_file_list_display(self) -> None:
        """Update the file list display by creating all items at once."""
        # Clear the list completely
        for widget in self.file_list.winfo_children():
            widget.destroy()

        # Reset our labels list
        self.image_labels = []

        # Show statistics if there are many files
        if len(self.filtered_image_paths) > 100:
            status_label = ctk.CTkLabel(
                self.file_list,
                text=f"{len(self.filtered_image_paths)} files",
                font=("Segoe UI", 10, "italic"),
                text_color="grey75"
            )
            status_label.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
            row_start = 1
        else:
            row_start = 0

        # Create all file items at once
        for i, file_path in enumerate(self.filtered_image_paths):
            label = self._create_file_item(i + row_start, file_path)
            self.image_labels.append(label)

        # Update scrollbar visibility
        self._update_scrollbar_visibility()

    def _update_file_list(self) -> None:
        """Initialize the file list with all image paths."""
        # Initialize filtered image paths
        self.filtered_image_paths = self.image_rel_paths.copy()

        # Reset scroll position
        if hasattr(self.file_list, "_parent_canvas") and self.file_list._parent_canvas.winfo_exists():
            self.file_list._parent_canvas.yview_moveto(0.0)

        # Update the display
        self._update_file_list_display()

    def _update_scrollbar_visibility(self) -> None:
        """Update scrollbar visibility based on file list content."""
        try:
            if hasattr(self.file_list, "_scrollbar_frame") and hasattr(
                self.file_list, "_parent_canvas"
            ):
                canvas = self.file_list._parent_canvas
                inner_frame = self.file_list._scrollbar_frame
                if inner_frame.winfo_reqheight() > canvas.winfo_height():
                    self._file_list_scrollbar.grid()
                else:
                    self._file_list_scrollbar.grid_remove()
            else:
                self._file_list_scrollbar.grid() if len(
                    self.image_rel_paths
                ) > 15 else self._file_list_scrollbar.grid_remove()
        except Exception as e:
            self._file_list_scrollbar.grid()
            logger.debug(f"Note: Could not adjust scrollbar visibility: {e}")

    def _restore_scroll_position(self, position: float) -> None:
        """Restore scroll position after updating file list."""
        with contextlib.suppress(Exception):
            self.file_list._parent_canvas.yview_moveto(position)

    def _update_scrollbar_visibility(self) -> None:
        """Update scrollbar visibility based on file list content."""
        try:
            if hasattr(self.file_list, "_scrollbar_frame") and hasattr(
                self.file_list, "_parent_canvas"
            ):
                canvas = self.file_list._parent_canvas
                inner_frame = self.file_list._scrollbar_frame
                if inner_frame.winfo_reqheight() > canvas.winfo_height():
                    self._file_list_scrollbar.grid()
                else:
                    self._file_list_scrollbar.grid_remove()
            else:
                self._file_list_scrollbar.grid() if len(
                    self.image_rel_paths
                ) > 15 else self._file_list_scrollbar.grid_remove()
        except Exception as e:
            self._file_list_scrollbar.grid()
            logger.debug(f"Note: Could not adjust scrollbar visibility: {e}")

    def _create_editor_panel(self, parent: ctk.CTkFrame) -> None:
        """Create the editor panel with tools, image container, and caption area."""
        editor_frame = ctk.CTkFrame(parent, fg_color=get_theme_color("main_frame_bg"))
        editor_frame.grid(row=0, column=1, sticky="nsew")
        editor_frame.grid_columnconfigure(0, weight=1)
        editor_frame.grid_rowconfigure(0, weight=0)
        editor_frame.grid_rowconfigure(1, weight=1)
        editor_frame.grid_rowconfigure(2, weight=0)
        self._create_tools_bar(editor_frame)
        self._create_image_container(editor_frame)
        self._create_caption_area(editor_frame)

    def _create_tools_bar(self, parent: ctk.CTkFrame) -> None:
        """Create the tools bar for mask editing."""
        tools_frame = ctk.CTkFrame(parent, fg_color=get_theme_color("tools_frame_bg"))
        tools_frame.grid(row=0, column=0, sticky="new")
        # Update the column count for the added button
        for i in range(10):
            tools_frame.grid_columnconfigure(i, weight=0)
        tools_frame.grid_columnconfigure(5, weight=1)
        icons = {
            "draw": "draw",
            "fill": "fill",
            "save": "save",
            "undo": "undo",
            "redo": "redo",
            "reset": "reset",
        }
        self._create_icon_button(
            tools_frame,
            0,
            0,
            "Draw",
            self.mask_editor.switch_to_brush_mode,
            icons["draw"],
            "Draw mask with brush",
        )
        self._create_icon_button(
            tools_frame,
            0,
            1,
            "Fill",
            self.mask_editor.switch_to_fill_mode,
            icons["fill"],
            "Fill areas of the mask",
        )
        ctk.CTkCheckBox(
            tools_frame,
            text="Edit Mode",
            variable=self.enable_mask_editing_var,
            command=self.mask_editor.update_cursor
        ).grid(row=0, column=2, padx=5, pady=2)
        self.brush_opacity_entry = ctk.CTkEntry(tools_frame, width=40)
        self.brush_opacity_entry.insert(0, "1.0")
        self.brush_opacity_entry.grid(row=0, column=3, padx=5, pady=2)
        self.brush_opacity_entry.bind("<FocusOut>", self._validate_brush_opacity)
        ctk.CTkLabel(tools_frame, text="Brush Opacity").grid(
            row=0, column=4, padx=2, pady=2
        )

        self._create_icon_button(
            tools_frame,
            0,
            6,
            "Clear",
            self.file_manager.reset_current_image,
            icons["reset"],
            "Reset image (clear caption and mask, save to commit change)",
        )

        self._create_icon_button(
            tools_frame,
            0,
            7,
            "",
            self.file_manager.save_changes,
            icons["save"],
            "Save changes",
        )

        self._create_icon_button(
            tools_frame,
            0,
            8,
            "",
            lambda: self.mask_editor.undo_mask_edit(),
            icons["undo"],
            "Undo last edit",
        )

        self._create_icon_button(
            tools_frame,
            0,
            9,
            "",
            lambda: self.mask_editor.redo_mask_edit(),
            icons["redo"],
            "Redo last undone edit",
        )

    def _create_image_container(self, parent: ctk.CTkFrame) -> None:
        """Create the image display container."""
        self.image_container = ctk.CTkFrame(parent, fg_color=get_theme_color("image_container_bg"))
        self.image_container.grid(row=1, column=0, sticky="nsew")
        self.image_container.grid_columnconfigure(0, weight=1)
        self.image_container.grid_rowconfigure(0, weight=1)
        self.image_container.grid_propagate(False)
        placeholder: Image.Image = Image.new(
            "RGB", (DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT), get_theme_color("canvas_bg")
        )
        self.image_tk: ImageTk.PhotoImage = ImageTk.PhotoImage(placeholder)
        self.image_label = ctk.CTkLabel(
            self.image_container, text="", image=self.image_tk
        )
        self.image_label.grid(row=0, column=0, sticky="nsew")
        # Bind events for mask editing and resizing.
        self.image_label.bind(
            "<Motion>", self.mask_editor.handle_mask_edit
        )
        self.image_label.bind(
            "<Button-1>", self.mask_editor.handle_mask_edit_start
        )
        self.image_label.bind(
            "<ButtonRelease-1>", self.mask_editor.handle_mask_edit_end
        )
        self.image_label.bind(
            "<Button-3>", self.mask_editor.handle_mask_edit_start
        )
        self.image_label.bind(
            "<ButtonRelease-3>", self.mask_editor.handle_mask_edit_end
        )
        # Add binding for middle mouse button for reset
        self.image_label.bind("<Button-2>", self._handle_middle_click)

        # Make the label focusable for click interactions
        if "!label" in self.image_label.children:
            self.image_label.children["!label"].configure(takefocus=1)
        else:
            self.image_label.configure(takefocus=1)

        self.bind("<Configure>", self.image_handler.on_resize)
        bind_mousewheel(
            self.image_label,
            {self.image_label.children["!label"]},
            self.mask_editor.adjust_brush_size,
        )
        self.image_label.bind("<Enter>", self.mask_editor.update_cursor)
        self.image_label.bind("<Leave>", self.mask_editor.set_default_cursor)

    def _handle_middle_click(self, event: tk.Event) -> None:
        """Handle middle mouse button click as a shortcut for reset."""
        self.file_manager.reset_current_image()
        return "break"

    def _create_caption_area(self, parent: ctk.CTkFrame) -> None:
        """Create the caption area for editing."""
        caption_frame = ctk.CTkFrame(parent, fg_color=get_theme_color("caption_frame_bg"))
        caption_frame.grid(row=2, column=0, sticky="sew")
        caption_frame.grid_columnconfigure(0, weight=0)
        caption_frame.grid_columnconfigure(1, weight=1)
        self.caption_line_values: list[str] = [
            f"Caption {i}" for i in range(1, 6)
        ]
        self.caption_line_var = ctk.StringVar(
            value=self.caption_line_values[0]
        )
        ctk.CTkOptionMenu(
            caption_frame,
            values=self.caption_line_values,
            variable=self.caption_line_var,
            command=self.caption_manager.on_caption_line_changed,
        ).grid(row=0, column=0, padx=5, pady=5)
        self.caption_entry = ctk.CTkEntry(caption_frame)
        self.caption_entry.grid(
            row=0, column=1, padx=5, pady=5, sticky="ew"
        )
        # Bind to store value on focus out
        self.caption_entry.bind("<FocusOut>", self._on_caption_focus_out)
        self._bind_key_events(self.caption_entry)

    def _on_caption_focus_out(self, event: tk.Event | None = None) -> None:
        """Store caption value when focus changes."""
        current_text: str = self.caption_entry.get()
        self.caption_lines[self.current_caption_line] = current_text

    def _bind_key_events(self, component: ctk.CTkBaseClass) -> None:
        """Bind common key events to the given component."""
        self.bind("<Right>", self.navigation_manager.next_image)
        self.bind("<Left>", self.navigation_manager.previous_image)
        component.bind("<Return>", self.file_manager.save_changes)
        self.bind("<Tab>", self.caption_manager.next_caption_line)
        self.bind("<Control-m>", self.mask_editor.toggle_mask_visibility_mode)
        self.bind("<Control-d>", self.mask_editor.switch_to_brush_mode)
        self.bind("<Control-f>", self.mask_editor.switch_to_fill_mode)
        self.bind("<Control-s>", self.file_manager.save_changes)
        self.bind("<bracketleft>", self.mask_editor.stepped_decrease_brush_size)
        self.bind("<bracketright>", self.mask_editor.stepped_increase_brush_size)
        self.bind("<Control-z>", self.mask_editor.undo_mask_edit)
        self.bind("<Control-y>", self.mask_editor.redo_mask_edit)

        # Add support for removing focus from entry fields when clicking elsewhere
        self.bind_all("<Button-1>", self._clear_focus, add="+")

    def _clear_focus(self, event: tk.Event) -> None:
        """Clear focus from input widgets when clicking elsewhere."""
        # Skip if already focusing on the clicked widget
        if event.widget == self.focus_get():
            return

        # Focus on the main window to remove focus from input widgets
        focused = self.focus_get()
        if focused and str(focused).endswith("entry"):
            # If we're losing focus from an entry, save its content first
            if focused == self.caption_entry:
                self._on_caption_focus_out(None)
            elif focused == self.brush_opacity_entry:
                self._validate_brush_opacity(None)

            # Set focus to clicked widget or main window
            self.focus_set()

    def _validate_brush_opacity(self, event: tk.Event | None = None) -> None:
        """Validate brush opacity value when focus changes."""
        try:
            opacity = float(self.brush_opacity_entry.get())
            # Ensure value is between 0 and 1
            opacity = max(0.0, min(1.0, opacity))
            self.brush_opacity_entry.delete(0, "end")
            self.brush_opacity_entry.insert(0, f"{opacity:.1f}")
        except (ValueError, TypeError):
            # Reset to default on invalid input
            self.brush_opacity_entry.delete(0, "end")
            self.brush_opacity_entry.insert(0, "1.0")

    def refresh_ui(self) -> None:
        """Refresh the image and caption UI."""
        if self.pil_image:
            self.image_handler.refresh_image()
            self.caption_manager.refresh_caption()

    def clear_ui(self) -> None:
        """Clear the current image, mask, and caption data."""
        if self.image_container and self.image_container.winfo_ismapped():
            width = max(MIN_CONTAINER_SIZE, self.image_container.winfo_width())
            height = max(MIN_CONTAINER_SIZE, self.image_container.winfo_height())
        else:
            width, height = DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT
        empty_image: Image.Image = Image.new(
            "RGB", (width, height), get_theme_color("canvas_bg")
        )
        self.image_tk = ImageTk.PhotoImage(empty_image)
        self.image_label.configure(image=self.image_tk)
        self.caption_entry.delete(0, "end")
        self.pil_image = None
        self.pil_mask = None
        self.caption_lines = [""] * 5
        self.current_caption_line = 0
        self.caption_line_var.set(self.caption_line_values[0])

    def _open_caption_window(self) -> None:
        """Open the auto-caption generation window as a modal dialog."""
        if self.dir:
            dialog = GenerateCaptionsWindow(
                self,
                self.dir,
                self.config_ui_data["include_subdirectories"],
            )
            self.wait_window(dialog)
            if 0 <= self.current_image_index < len(self.image_rel_paths):
                self.navigation_manager.switch_to_image(self.current_image_index)

    def _open_mask_window(self) -> None:
        """Open the auto-mask generation window as a modal dialog."""
        if self.dir:
            dialog = GenerateMasksWindow(
                self,
                self.dir,
                self.config_ui_data["include_subdirectories"],
            )
            self.wait_window(dialog)
            if 0 <= self.current_image_index < len(self.image_rel_paths):
                self.navigation_manager.switch_to_image(self.current_image_index)

    def _open_file_tools(self) -> None:
        """Open the file tools window as a modal dialog."""
        file_ops_window = FileOperationsWindow(self, self.dir)
        self.wait_window(file_ops_window)
        if 0 <= self.current_image_index < len(self.image_rel_paths):
            self.navigation_manager.switch_to_image(self.current_image_index)

    def _open_bulk_caption_edit(self) -> None:
        """Open the bulk caption edit window as a modal dialog."""
        if self.dir:
            dialog = BulkCaptionEditWindow(
                self,
                self.dir,
                self.config_ui_data["include_subdirectories"],
            )
            self.wait_window(dialog)
            if 0 <= self.current_image_index < len(self.image_rel_paths):
                self.navigation_manager.switch_to_image(self.current_image_index)

    def _post_dialog_focus_restore(self, previous_focus):
        """Restore focus after a dialog closes"""
        try:
            self.focus_force()
            self.lift()
            if previous_focus and previous_focus.winfo_exists():
                previous_focus.focus_set()
        except (tk.TclError, RuntimeError, AttributeError):
            # Handle cases where the window or widgets might no longer exist
            pass

    def _show_help(self) -> None:
        """Show help text (currently printed to console)."""
        logger.info(self.help_text)


class ImageHandler:
    def __init__(self, parent: CaptionUI) -> None:
        self.parent: CaptionUI = parent

    def _calculate_display_dimensions(self) -> tuple[int, int, int, int]:
        """
        Calculate display dimensions to fit the image in the container while preserving aspect ratio.
        Returns (display_width, display_height, left_offset, top_offset).
        """
        container_width: int = max(
            MIN_CONTAINER_SIZE, self.parent.image_container.winfo_width()
        )
        container_height: int = max(
            MIN_CONTAINER_SIZE, self.parent.image_container.winfo_height()
        )
        if container_width <= MIN_CONTAINER_SIZE:
            container_width = DEFAULT_IMAGE_WIDTH
        if container_height <= MIN_CONTAINER_SIZE:
            container_height = DEFAULT_IMAGE_HEIGHT

        image_width: int = self.parent.image_width
        image_height: int = self.parent.image_height
        if image_width <= 0 or image_height <= 0:
            return 0, 0, 0, 0

        padding: int = IMAGE_PADDING
        available_width: int = container_width - (padding * 2)
        available_height: int = container_height - (padding * 2)
        scale: float = min(
            available_width / image_width, available_height / image_height
        )
        scale = min(scale, 1.0)
        display_width: int = int(image_width * scale)
        display_height: int = int(image_height * scale)
        left_offset: int = (container_width - display_width) // 2
        top_offset: int = (container_height - display_height) // 2

        return display_width, display_height, left_offset, top_offset

    def refresh_image(self) -> None:
        """Refresh and display the current image (with mask, if available) in the container."""
        if not self.parent.pil_image:
            return

        # Get dimensions and create canvas
        display_dimensions = self._prepare_display()
        if not all(display_dimensions):
            return

        # Prepare the image and optional mask
        final_image = self._prepare_image_with_mask(display_dimensions)

        # Update the display
        self._update_display(final_image, display_dimensions)

    def _prepare_display(self) -> tuple[int, int, int, int]:
        """Prepare display dimensions and store in parent."""
        display_width, display_height, left_offset, top_offset = self._calculate_display_dimensions()

        # Store values in parent for use by other methods
        self.parent.display_width = display_width
        self.parent.display_height = display_height
        self.parent.left_offset = left_offset
        self.parent.top_offset = top_offset

        return display_width, display_height, left_offset, top_offset

    def _prepare_image_with_mask(self, dimensions: tuple[int, int, int, int]) -> Image.Image:
        """Create the final image with mask applied if available."""
        display_width, display_height, left_offset, top_offset = dimensions
        container_width = self.parent.image_container.winfo_width()
        container_height = self.parent.image_container.winfo_height()

        # Create a blank canvas for the final image
        canvas = Image.new("RGB", (container_width, container_height), get_theme_color("image_container_bg"))

        # Get resized image
        resized_image = self._resize_image(display_width, display_height)

        # If mask exists, process it
        if self.parent.pil_mask:
            final_image = self._process_masked_image(resized_image, display_width, display_height)
        else:
            final_image = resized_image

        # Paste onto the canvas
        canvas.paste(final_image, (left_offset, top_offset))
        return canvas

    def _resize_image(self, width: int, height: int) -> Image.Image:
        """Resize the current image to specified dimensions."""
        if width and height:
            return self.parent.pil_image.resize((width, height), Image.Resampling.LANCZOS)
        return self.parent.pil_image.copy()

    def _process_masked_image(self, resized_image: Image.Image, width: int, height: int) -> Image.Image:
        """Process image with mask and return the final image."""
        # Resize mask to match image
        resized_mask = self.parent.pil_mask.resize((width, height), Image.Resampling.NEAREST)

        # If display_only_mask is true, return just the mask
        if self.parent.mask_editor.display_only_mask:
            return resized_mask

        # Otherwise blend mask with image
        return self._blend_mask_with_image(resized_image, resized_mask)

    def _blend_mask_with_image(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        """Blend a mask with an image according to mask opacity settings."""

        np_image = np.array(image, dtype=np.float32) / 255.0
        np_mask = np.array(mask, dtype=np.float32) / 255.0

        # Apply minimum opacity to ensure mask is visible
        np_mask = self._adjust_mask_opacity(np_mask)

        # Apply mask to image
        np_result = (np_image * np_mask * 255.0).astype(np.uint8)
        return Image.fromarray(np_result, mode="RGB")

    def _adjust_mask_opacity(self, np_mask: np.ndarray) -> np.ndarray:
        """Adjust mask opacity to ensure visibility."""
        min_opacity = MASK_MIN_OPACITY

        if np.min(np_mask) == 0:
            # Ensure completely masked areas have minimum opacity
            return np_mask * (1.0 - min_opacity) + min_opacity
        elif np.min(np_mask) < 1:
            # Scale mask values between min_opacity and 1.0
            min_val = float(np.min(np_mask))
            return (np_mask - min_val) / (1.0 - min_val) * (1.0 - min_opacity) + min_opacity

        return np_mask

    def _update_display(self, final_image: Image.Image, dimensions: tuple) -> None:
        """Update the Tkinter display with the processed image."""
        self.parent.image_tk = ImageTk.PhotoImage(final_image)
        self.parent.image_label.configure(image=self.parent.image_tk)

    def update_image_container_size(self) -> None:
        """Update container size and refresh image display."""
        if (
            self.parent.image_container.winfo_width() > MIN_CONTAINER_SIZE
            and self.parent.image_container.winfo_height() > MIN_CONTAINER_SIZE
        ):
            self.refresh_image()
        else:
            self.parent.image_container.config(width=DEFAULT_IMAGE_WIDTH, height=DEFAULT_IMAGE_HEIGHT)
            self.parent.after(100, self.update_image_container_size)

    def on_resize(self, event: tk.Event | None = None) -> None:
        """Handle window resize events to refresh the image display."""
        if (
            self.parent.pil_image
            and self.parent.image_container.winfo_ismapped()
        ):
            if hasattr(self.parent, "_resize_after"):
                self.parent.after_cancel(self.parent._resize_after)
            self.parent._resize_after = self.parent.after(
                100, self.refresh_image
            )

    def load_image_data(self) -> None:
        """Load image and mask data from the current file."""
        if not (
            0
            <= self.parent.current_image_index
            < len(self.parent.image_rel_paths)
        ):
            return

        # Use pathlib for path operations
        image_path = Path(self.parent.dir) / self.parent.image_rel_paths[self.parent.current_image_index]

        try:
            self.parent.pil_image = Image.open(image_path).convert("RGB")
            self.parent.image_width = self.parent.pil_image.width
            self.parent.image_height = self.parent.pil_image.height
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            self.parent.pil_image = None

        mask_path = image_path.with_name(f"{image_path.stem}-masklabel.png")
        if mask_path.exists():
            try:
                self.parent.pil_mask = Image.open(mask_path).convert("RGB")
                self.parent.mask_editor.reset_for_new_image()
            except Exception as e:
                logger.error(f"Error loading mask {mask_path}: {e}")
                self.parent.pil_mask = None
        else:
            self.parent.pil_mask = None
            self.parent.mask_editor.reset_for_new_image()

        caption_path = image_path.with_suffix(".txt")
        if caption_path.exists():
            try:
                content = caption_path.read_text(encoding="utf-8").strip()
                self.parent.caption_lines = content.split("\n")
                self.parent.caption_lines.extend(
                    [""] * (5 - len(self.parent.caption_lines))
                )
                self.parent.caption_lines = self.parent.caption_lines[:5]
            except Exception as e:
                logger.error(f"Error loading caption {caption_path}: {e}")
                self.parent.caption_lines = [""] * 5
        else:
            self.parent.caption_lines = [""] * 5

        self.parent.current_caption_line = 0
        self.parent.caption_line_var.set(
            self.parent.caption_line_values[0]
        )
        self.parent.caption_entry.delete(0, "end")
        self.parent.caption_entry.insert(0, self.parent.caption_lines[0])

    def is_supported_image(self, filename: str) -> bool:
        """Determine if the file is a supported image."""
        path = Path(filename)
        return (path_util.is_supported_image_extension(path.suffix) and
                not path.stem.endswith("-masklabel"))


class MaskEditor:

    def __init__(self, parent: CaptionUI) -> None:
        self.parent: CaptionUI = parent
        self.mask_draw_x: float = 0.0
        self.mask_draw_y: float = 0.0
        self.mask_draw_radius: float = DEFAULT_BRUSH_SIZE
        self.mask_editing_mode: EditMode = EditMode.DRAW
        self.display_only_mask: bool = False
        self.mask_history: list[Image.Image] = []
        self.mask_history_position: int = -1
        self.mask_history_limit: int = MASK_HISTORY_LIMIT
        self.is_editing: bool = False
        self.edit_started: bool = False
        self._cached_image_dimensions: tuple[int, int] = (0, 0)

    def stepped_decrease_brush_size(self, event: tk.Event | None = None) -> str:
        """Decrease the brush size by a fixed step."""
        self.mask_draw_radius = max(MIN_BRUSH_SIZE, self.mask_draw_radius - BRACKET_BRUSH_SIZE_STEP)
        return "break"

    def stepped_increase_brush_size(self, event: tk.Event | None = None) -> str:
        """Increase the brush size by a fixed step."""
        self.mask_draw_radius = min(MAX_BRUSH_SIZE, self.mask_draw_radius + BRACKET_BRUSH_SIZE_STEP)
        return "break"

    def reset_for_new_image(self) -> None:
        """Reset mask editing state for a new image."""
        self.mask_history = []
        self.mask_history_position = -1
        self.is_editing = False
        self.edit_started = False
        self._cached_image_dimensions = (
            self.parent.image_width,
            self.parent.image_height,
        )
        if self.parent.pil_mask:
            self.mask_history.append(self.parent.pil_mask.copy())
            self.mask_history_position = 0

    def handle_mask_edit_start(self, event: tk.Event) -> None:
        """Start a mask edit action."""
        if not self._can_edit_mask(event):
            return
        self.is_editing = True
        self.edit_started = False
        self.handle_mask_edit(event)

    def handle_mask_edit_end(self, event: tk.Event) -> None:
        """End a mask edit action."""
        if not self.is_editing:
            return
        self.is_editing = False
        self.handle_mask_edit(event)
        if self.edit_started:
            self._save_mask_to_history()
            self.edit_started = False

    def handle_mask_edit(self, event: tk.Event) -> None:
        """Handle mask editing events."""
        if not self._can_edit_mask(event):
            return

        # Get coordinates for editing
        coordinates = self._convert_screen_to_mask_coordinates(event)
        start_x, start_y, end_x, end_y = coordinates

        if start_x == end_x == 0 and start_y == end_y == 0:
            return

        # Determine which mouse button is pressed
        mouse_buttons = self._determine_mouse_buttons(event)
        is_left, is_right = mouse_buttons

        if not (is_left or is_right) and not self.is_editing:
            return

        # Apply appropriate editing action
        self._apply_editing_action(start_x, start_y, end_x, end_y, is_left, is_right)

    def _can_edit_mask(self, event: tk.Event) -> bool:
        """Check if mask editing is permitted."""
        return (
            self.parent.enable_mask_editing_var.get()
            and event.widget
            == self.parent.image_label.children.get("!label", event.widget)
            and self.parent.pil_image is not None
            and 0
            <= self.parent.current_image_index
            < len(self.parent.image_rel_paths)
        )

    def _convert_screen_to_mask_coordinates(self, event: tk.Event) -> ImageCoordinates:
        """Convert screen coordinates to mask coordinates."""
        # Get event coordinates
        event_position = self._get_event_position(event)
        if not event_position:
            return 0, 0, 0, 0

        # Convert to image coordinates
        start_coordinates = self._convert_to_image_coordinates(event_position)
        if not start_coordinates:
            return 0, 0, 0, 0

        # Calculate end coordinates for line drawing
        end_coordinates = self._get_end_coordinates(event_position)

        # Store current position for next call
        self._store_current_position(event)

        # Return start_x, start_y, end_x, end_y
        return start_coordinates[0], start_coordinates[1], end_coordinates[0], end_coordinates[1]

    def _get_event_position(self, event: tk.Event) -> tuple[float, float] | None:
        """Extract event position and check if it's within the image area."""
        event_x = event.x
        event_y = event.y
        return event_x, event_y

    def _convert_to_image_coordinates(self, position: tuple[float, float]) -> tuple[int, int] | None:
        """Convert screen coordinates to image coordinates."""
        event_x, event_y = position
        left_offset = self.parent.left_offset
        top_offset = self.parent.top_offset
        display_width = self.parent.display_width
        display_height = self.parent.display_height

        # Calculate position relative to image area
        image_x = event_x - left_offset
        image_y = event_y - top_offset

        # Check if coordinates are within the image area
        if not (0 <= image_x < display_width and 0 <= image_y < display_height):
            return None

        # Convert to original image coordinates
        image_width = self.parent.image_width
        image_height = self.parent.image_height

        start_x = int(image_x * image_width / display_width)
        start_y = int(image_y * image_height / display_height)

        return start_x, start_y

    def _get_end_coordinates(self, current_position: tuple[float, float]) -> tuple[int, int]:
        """Calculate end coordinates based on previous position for line drawing."""
        event_x, event_y = current_position
        left_offset = self.parent.left_offset
        top_offset = self.parent.top_offset
        display_width = self.parent.display_width
        display_height = self.parent.display_height
        image_width = self.parent.image_width
        image_height = self.parent.image_height

        # Use previous position if available, otherwise use current position
        if hasattr(self, "mask_draw_x") and hasattr(self, "mask_draw_y"):
            prev_image_x = self.mask_draw_x - left_offset
            prev_image_y = self.mask_draw_y - top_offset

            if 0 <= prev_image_x < display_width and 0 <= prev_image_y < display_height:
                end_x = int(prev_image_x * image_width / display_width)
                end_y = int(prev_image_y * image_height / display_height)
                return end_x, end_y

        # If no previous position or out of bounds, use the starting position
        start_coords = self._convert_to_image_coordinates(current_position)
        if start_coords:
            return start_coords
        return 0, 0

    def _store_current_position(self, event: tk.Event) -> None:
        """Store current mouse position for next event processing."""
        self.mask_draw_x = event.x
        self.mask_draw_y = event.y

    def _determine_mouse_buttons(self, event: tk.Event) -> tuple[bool, bool]:
        """Determine which mouse buttons are pressed."""
        is_left = bool(event.state & 0x0100 or event.num == 1)
        is_right = bool(event.state & 0x0400 or event.num == 3)
        return is_left, is_right

    def _apply_editing_action(
        self,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        is_left: bool,
        is_right: bool
    ) -> None:
        """Apply the appropriate mask editing action based on current mode."""
        if self.mask_editing_mode == EditMode.DRAW:
            self._draw_mask(start_x, start_y, end_x, end_y, is_left, is_right)
        elif self.mask_editing_mode == EditMode.FILL:
            self._fill_mask(start_x, start_y, is_left, is_right)

    def _determine_brush_mask_color(self, is_left: bool) -> RGBColor | None:
        """Determine the brush color based (b or w) base on the mouse button."""
        if is_left:
            try:
                opacity: float = float(
                    self.parent.brush_opacity_entry.get()
                )
                opacity = max(0.0, min(1.0, opacity))
            except (ValueError, TypeError):
                opacity = 1.0
            # Return black (masked) for left click
            return (0, 0, 0)
        # Return white (unmasked) for right click
        return (255, 255, 255)

    def _ensure_mask_exists(self, adding_to_mask: bool) -> None:
        """Ensure a mask image exists for editing."""
        if self.parent.pil_mask is None:
            color: tuple[int, int, int] = (
                (0, 0, 0) if adding_to_mask else (255, 255, 255)
            )
            self.parent.pil_mask = Image.new(
                "RGB",
                (self.parent.image_width, self.parent.image_height),
                color=color,
            )

    def _draw_mask(
        self,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        is_left: bool,
        is_right: bool,
    ) -> None:
        """Draw on the mask image."""
        color = self._determine_brush_mask_color(is_left)
        if not color:
            return

        # Ensure mask exists and prepare for editing
        self._ensure_mask_exists(is_left)
        if not self.edit_started:
            self._save_mask_to_history()
            self.edit_started = True

        # Calculate brush parameters
        radius = self._calculate_brush_radius()
        if radius <= 0 or (start_x == end_x == start_y == end_y == 0):
            return

        # Draw on the mask
        self._draw_on_mask(start_x, start_y, end_x, end_y, radius, color)

        # Update display
        self.parent.image_handler.refresh_image()

    def _calculate_brush_radius(self) -> int:
        """Calculate brush radius based on mask dimensions and radius setting, ensuring minimum 1px."""
        max_dimension = max(self.parent.pil_mask.width, self.parent.pil_mask.height)
        # Ensure radius is at least 1 pixel even with small brush sizes
        return max(1, int(self.mask_draw_radius * max_dimension))

    def _draw_on_mask(
        self,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        radius: int,
        color: RGBColor
    ) -> None:
        """Draw line and end points on mask."""
        draw = ImageDraw.Draw(self.parent.pil_mask)

        # Draw line between points
        line_width = 2 * radius + 1
        draw.line((start_x, start_y, end_x, end_y), fill=color, width=line_width)

        # Draw circle at start point
        draw.ellipse(
            (
                start_x - radius,
                start_y - radius,
                start_x + radius,
                start_y + radius,
            ),
            fill=color,
        )

        # Draw circle at end point if different from start
        if (start_x, start_y) != (end_x, end_y):
            draw.ellipse(
                (
                    end_x - radius,
                    end_y - radius,
                    end_x + radius,
                    end_y + radius,
                ),
                fill=color,
            )

    def _fill_mask(
        self, start_x: int, start_y: int, is_left: bool, is_right: bool
    ) -> None:
        """Fill an area of the mask image."""
        color: RGBColor | None = self._determine_brush_mask_color(is_left)
        if color:
            self._ensure_mask_exists(is_left)
            if not (
                0 <= start_x < self.parent.image_width
                and 0 <= start_y < self.parent.image_height
            ):
                return
            self._save_mask_to_history()
            self.edit_started = True
            np_mask: np.ndarray = np.array(
                self.parent.pil_mask, dtype=np.uint8
            )
            cv2.floodFill(np_mask, None, (start_x, start_y), color)
            self.parent.pil_mask = Image.fromarray(np_mask, "RGB")
            self.parent.image_handler.refresh_image()

    def adjust_brush_size(self, delta: float, raw_event: object) -> None:
        """Adjust the brush size based on mouse wheel movement, ensuring valid size."""
        multiplier: float = 1.0 + (
            delta * (BRUSH_SIZE_WHEEL_SMALL_FACTOR if self.mask_draw_radius < BRUSH_SIZE_THRESHOLD else BRUSH_SIZE_WHEEL_LARGE_FACTOR)
        )
        # Calculate new size and enforce minimum/maximum
        new_radius = self.mask_draw_radius * multiplier
        self.mask_draw_radius = max(MIN_BRUSH_SIZE, min(MAX_BRUSH_SIZE, new_radius))

    def _save_mask_to_history(self) -> None:
            """Save the current mask state to history for undo/redo."""
            if self.parent.pil_mask is None:
                return
            current_mask: Image.Image = self.parent.pil_mask.copy()
            if self.mask_history_position < len(self.mask_history) - 1:
                self.mask_history = self.mask_history[
                    : self.mask_history_position + 1
                ]
            self.mask_history.append(current_mask)
            if len(self.mask_history) > self.mask_history_limit:
                self.mask_history.pop(0)
            self.mask_history_position = len(self.mask_history) - 1

            # Mark the mask as modified for the current image
            if hasattr(self.parent.file_manager, '_mask_modified') and self.parent.dir and 0 <= self.parent.current_image_index < len(self.parent.image_rel_paths):
                image_path = Path(self.parent.dir) / self.parent.image_rel_paths[self.parent.current_image_index]
                self.parent.file_manager._mask_modified[str(image_path)] = True

    def undo_mask_edit(
        self, event: tk.Event | None = None
    ) -> str | None:
        """Undo the last mask edit."""
        if not self.mask_history or self.mask_history_position <= 0:
            return "break" if event else None
        self.mask_history_position -= 1
        self.parent.pil_mask = self.mask_history[
            self.mask_history_position
        ].copy()
        self.parent.image_handler.refresh_image()
        return "break" if event else None

    def redo_mask_edit(
        self, event: tk.Event | None = None
    ) -> str | None:
        """Redo the previously undone mask edit."""
        if self.mask_history_position >= len(self.mask_history) - 1:
            return "break" if event else None
        self.mask_history_position += 1
        self.parent.pil_mask = self.mask_history[
            self.mask_history_position
        ].copy()
        self.parent.image_handler.refresh_image()
        return "break" if event else None

    def switch_to_brush_mode(self, event: tk.Event | None = None) -> str | None:
        """Switch to draw mask mode."""
        self.mask_editing_mode = EditMode.DRAW
        self.update_cursor()
        return "break" if event else None

    def switch_to_fill_mode(self, event: tk.Event | None = None) -> str | None:
        """Switch to fill mask mode."""
        self.mask_editing_mode = EditMode.FILL
        self.update_cursor()
        return "break" if event else None

    def toggle_mask_visibility_mode(self, event: tk.Event | None = None) -> str:
        """Toggle between displaying only the mask or the combined image."""
        self.display_only_mask = not self.display_only_mask
        self.parent.image_handler.refresh_image()
        return "break"

    def update_cursor(self, event: tk.Event | None = None) -> None:
        """Update the cursor based on the current mask editing mode."""
        if not self.parent.enable_mask_editing_var.get():
            self.set_default_cursor()
            return

        if self.mask_editing_mode == EditMode.DRAW:
            self.set_brush_cursor()
        elif self.mask_editing_mode == EditMode.FILL:
            self.set_fill_cursor()

    def set_brush_cursor(self) -> None:
        """Set the cursor to brush mode."""
        brush_cursor = get_platform_cursor("brush", "pencil")
        if "!label" in self.parent.image_label.children:
            self.parent.image_label.children["!label"].configure(cursor=brush_cursor)
        else:
            self.parent.image_label.configure(cursor=brush_cursor)

    def set_fill_cursor(self) -> None:
        """Set the cursor to fill mode."""
        fill_cursor = get_platform_cursor("fill", "dotbox")
        if "!label" in self.parent.image_label.children:
            self.parent.image_label.children["!label"].configure(cursor=fill_cursor)
        else:
            self.parent.image_label.configure(cursor=fill_cursor)

    def set_default_cursor(self, event: tk.Event | None = None) -> None:
        """Reset to the default cursor."""
        if "!label" in self.parent.image_label.children:
            self.parent.image_label.children["!label"].configure(cursor="")
        else:
            self.parent.image_label.configure(cursor="")


class NavigationManager:
    def __init__(self, parent: CaptionUI) -> None:
        self.parent: CaptionUI = parent
        self.last_highlighted_label = None  # Keep track of the last highlighted label

    def switch_to_image(self, index: int, from_click: bool = False) -> None:
        """Switch to the image at the given index."""
        # Clear highlight from the last highlighted item
        if self.last_highlighted_label and self.last_highlighted_label.winfo_exists():
            with contextlib.suppress(tk.TclError, RuntimeError):
                self.last_highlighted_label.configure(
                    text_color=ThemeManager.theme["CTkLabel"]["text_color"],
                    fg_color="transparent",
                    corner_radius=0,
                )

        # Update current index
        self.parent.current_image_index = index

        # Only ensure visibility if not clicked (keyboard navigation)
        if not from_click:
            self._ensure_selected_item_visible()

        # Highlight the new item if it exists in the filtered list
        if 0 <= index < len(self.parent.image_rel_paths):
            try:
                # Find the new item in the filtered paths
                new_path = self.parent.image_rel_paths[index]
                if new_path in self.parent.filtered_image_paths:
                    filtered_idx = self.parent.filtered_image_paths.index(new_path)
                    if filtered_idx < len(self.parent.image_labels):
                        new_label = self.parent.image_labels[filtered_idx]
                        new_label.configure(
                            text_color=get_theme_color("selected_file_text"),
                            fg_color=get_theme_color("selected_file_bg"),
                            corner_radius=6,
                        )
                        # Store reference to the highlighted label
                        self.last_highlighted_label = new_label
            except (ValueError, IndexError):
                pass

        # Load and display the image regardless of whether the label exists
        if 0 <= index < len(self.parent.image_rel_paths):
            self.parent.image_handler.load_image_data()
            self.parent.refresh_ui()
        else:
            self.parent.clear_ui()

    def _ensure_selected_item_visible(self) -> None:
        """Make sure the selected item is visible in the scrollable area."""
        if not (0 <= self.parent.current_image_index < len(self.parent.image_rel_paths)):
            return

        # Check if the current image is in the filtered list
        current_path = self.parent.image_rel_paths[self.parent.current_image_index]
        if current_path not in self.parent.filtered_image_paths:
            return

        # If we have a valid canvas, scroll to the item
        if (hasattr(self.parent.file_list, "_parent_canvas") and
                self.parent.file_list._parent_canvas.winfo_exists()):

            # Find the position in the filtered list
            try:
                filtered_idx = self.parent.filtered_image_paths.index(current_path)
                total_items = len(self.parent.filtered_image_paths)

                if total_items > 0:
                    # Calculate position (with padding) and scroll there
                    yview_position = filtered_idx / total_items
                    self.parent.file_list._parent_canvas.yview_moveto(
                        max(0.0, min(1.0, yview_position - 0.2))
                    )
            except (ValueError, IndexError):
                pass

    def next_image(self, event: tk.Event | None = None) -> str:
        """Switch to the next image."""
        if self.parent.image_rel_paths and (
            self.parent.current_image_index + 1
        ) < len(self.parent.image_rel_paths):
            self.parent.file_manager.save_changes()
            self.switch_to_image(self.parent.current_image_index + 1, from_click=False)
        return "break"

    def previous_image(self, event: tk.Event | None = None) -> str:
        """Switch to the previous image."""
        if (
            self.parent.image_rel_paths
            and (self.parent.current_image_index - 1) >= 0
        ):
            self.parent.file_manager.save_changes()
            self.switch_to_image(self.parent.current_image_index - 1, from_click=False)
        return "break"


class ModelManager:
    def __init__(
        self,
        device: torch.device | None = None,
        precision: torch.dtype | None = None,
    ) -> None:
        self.device: torch.device = device or default_device
        self.precision: torch.dtype = precision or torch.float16

        # Model registries for better extensibility
        self._captioning_registry = {
            "Moondream 2": Moondream2Model,
            "Blip2": Blip2Model, #TODO Replace with blip 3 when it comes out.
            "WD14 VIT v2": BooruModels,
            "WD EVA02-Large Tagger v3": BooruModels,
            "WD SwinV2 Tagger v3": BooruModels,
            "JoyTag": BooruModels,
        }

        self._masking_registry = {
            "ClipSeg": ClipSegModel,
            "Rembg": RembgModel,
            "Rembg-Human": RembgHumanModel,
            "Hex Color": MaskByColor,
            "SAMoondream": MoondreamSAMMaskModel,
        }

        self.masking_model = None
        self.captioning_model = None
        self.current_masking_model_name = None
        self.current_captioning_model_name = None

    def get_available_captioning_models(self) -> list[str]:
        """Return a list of available captioning models."""
        return list(self._captioning_registry.keys())

    def get_available_masking_models(self) -> list[str]:
        """Return a list of available masking models."""
        return list(self._masking_registry.keys())

    def load_masking_model(
            self, model: str
        ) -> ClipSegModel | RembgModel | RembgHumanModel | MaskByColor | MoondreamSAMMaskModel | None:
            """Load the specified masking model, unloading any captioning model."""
            self.captioning_model = None
            self.current_captioning_model_name = None

            if model not in self._masking_registry:
                logger.error(f"Unknown masking model: {model}")
                return None

            # If the requested model is already loaded, return it
            if self.current_masking_model_name == model and self.masking_model is not None:
                logger.info(f"Model {model} is already loaded")
                return self.masking_model

            model_class = self._masking_registry[model]

            if self.masking_model is None or not isinstance(
                self.masking_model, model_class
            ):
                logger.info(f"Loading {model} model, this may take a while")

                # Special handling for SAMoondream model
                if model == "SAMoondream":
                    logger.debug("Creating SAMoondream model with default parameters")
                    self.masking_model = model_class(
                        device=self.device,
                        dtype=torch.float32,
                        sam2_model_size="large",
                        moondream_model_revision="2025-01-09"  # Use latest Moondream version
                    )
                else:
                    # Default initialization for other models
                    self.masking_model = model_class(self.device, torch.float32)

                self.current_masking_model_name = model

            return self.masking_model

    def load_captioning_model(
        self, model: str, **kwargs
    ) -> Blip2Model | BooruModels | Moondream2Model | None:
        """Load the specified captioning model, unloading any masking model."""
        self.masking_model = None
        self.current_masking_model_name = None

        if model not in self._captioning_registry:
            logger.error(f"Unknown captioning model: {model}")
            return None

        # If the requested model is already loaded, return it
        if (
            self.current_captioning_model_name == model
            and self.captioning_model is not None
        ):
            logger.info(f"Model {model} is already loaded")
            return self.captioning_model

        model_class = self._captioning_registry[model]

        if self.captioning_model is None or not isinstance(
            self.captioning_model, model_class
        ):
            logger.info(f"Loading {model} model, this may take a while")

            # For BooruModels, pass the model name
            if model_class == BooruModels:
                logger.debug(f"Creating BooruModels with model_name='{model}'")
                self.captioning_model = model_class(
                    self.device, self.precision, model_name=model, **kwargs
                )
            else:
                logger.debug(f"Creating {model} with kwargs: {kwargs}")
                # Pass kwargs to all model types
                self.captioning_model = model_class(
                    self.device, self.precision, **kwargs
                )
            self.current_captioning_model_name = model

        return self.captioning_model

    def get_masking_model(
        self,
    ) -> ClipSegModel | RembgModel | RembgHumanModel | MaskByColor | None:
        return self.masking_model

    def get_captioning_model(
        self,
    ) -> Blip2Model | BooruModels | None:
        return self.captioning_model


class FileManager:
    def __init__(self, parent: CaptionUI) -> None:
        self.parent: CaptionUI = parent
        self._last_saved_caption = {}  # Simple cache to track saved content
        self._mask_modified = {}  # Track if mask has been modified
        self._reset_pending = {}  # Track images with pending reset

    def reset_current_image(self) -> None:
        """Reset the current image by clearing caption and mask (preview only until saved)."""
        if not (0 <= self.parent.current_image_index < len(self.parent.image_rel_paths)):
            return

        # Get the image path for the current image
        image_path = Path(self.parent.dir) / self.parent.image_rel_paths[self.parent.current_image_index]
        image_key = str(image_path)

        # Mark this image as pending reset
        self._reset_pending[image_key] = True

        # Clear the caption
        self.parent.caption_lines = [""] * 5
        self.parent.caption_entry.delete(0, "end")

        # Clear the mask
        self.parent.pil_mask = None
        self.parent.mask_editor.reset_for_new_image()

        # Refresh the UI to show the preview
        self.parent.refresh_ui()

        # Show message to user
        logger.info("Reset preview applied. Press Ctrl+S or click Save to confirm and delete files.")

    def save_changes(self, event: tk.Event | None = None) -> None:
        """Save the current mask and caption data to disk."""
        if not (
            0
            <= self.parent.current_image_index
            < len(self.parent.image_rel_paths)
        ):
            return

        # Use pathlib for path operations
        image_path = (
            Path(self.parent.dir)
            / self.parent.image_rel_paths[self.parent.current_image_index]
        )
        image_key = str(image_path)

        # Check if this image is pending reset
        if self._reset_pending.get(image_key, False):
            # Actually delete the files
            mask_path = image_path.with_name(f"{image_path.stem}-masklabel.png")
            caption_path = image_path.with_suffix(".txt")

            # Delete mask file if it exists
            if mask_path.exists():
                try:
                    mask_path.unlink()
                    logger.info(f"Deleted mask file: {mask_path}")
                except Exception as e:
                    logger.error(f"Error deleting mask file: {e}")

            # Delete caption file if it exists
            if caption_path.exists():
                try:
                    caption_path.unlink()
                    logger.info(f"Deleted caption file: {caption_path}")
                except Exception as e:
                    logger.error(f"Error deleting caption file: {e}")

            # Clear the pending reset flag
            self._reset_pending[image_key] = False

            # Clear the saved caption cache
            if image_key in self._last_saved_caption:
                del self._last_saved_caption[image_key]

            # Clear the modified mask flag
            if image_key in self._mask_modified:
                del self._mask_modified[image_key]

            return

        # Handle mask saving if it exists and was modified
        if self.parent.pil_mask:
            mask_path = image_path.with_name(
                f"{image_path.stem}-masklabel.png"
            )
            if self._mask_modified.get(image_key, False):
                try:
                    self.parent.pil_mask.save(mask_path)
                    logger.info(f"Saved mask to {mask_path}")
                    self._mask_modified[image_key] = False
                except Exception as e:
                    logger.error(f"Error saving mask: {e}")

        # Handle caption saving with change detection
        current_text = self.parent.caption_entry.get()
        self.parent.caption_lines[self.parent.current_caption_line] = (
            current_text
        )

        non_empty_lines = [
            line for line in self.parent.caption_lines if line
        ]
        caption_content = "\n".join(non_empty_lines)
        caption_path = image_path.with_suffix(".txt")

        # Only save if content has changed
        if self._last_saved_caption.get(image_key) != caption_content:
            # Double-check against file content if file exists
            should_save = True
            if caption_path.exists():
                try:
                    file_content = caption_path.read_text(
                        encoding="utf-8"
                    ).strip()
                    should_save = file_content != caption_content
                except Exception:
                    # If reading fails, we'll save anyway
                    pass

            if should_save:
                try:
                    caption_path.write_text(
                        caption_content, encoding="utf-8"
                    )
                    logger.info(f"Saved caption to {caption_path}")
                    self._last_saved_caption[image_key] = caption_content
                except Exception as e:
                    logger.error(f"Error saving caption: {e}")

    def open_in_explorer(self) -> None:
        """Open the current image location in the system file explorer."""
        if not (
            0 <= self.parent.current_image_index < len(self.parent.image_rel_paths)
        ):
            return

        image_path = Path(self.parent.dir) / self.parent.image_rel_paths[self.parent.current_image_index]
        image_path = image_path.resolve()  # Gets absolute normalized path

        try:
            if platform.system() == "Windows":
                subprocess.run(["explorer", f"/select,{image_path}"], check=False)
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(["open", "-R", str(image_path)], check=False)
            else:  # Linux
                subprocess.run(["xdg-open", str(image_path.parent)], check=False)
        except Exception as e:
            logger.error(f"Error opening file in explorer: {e}")

    def open_directory(self) -> None:
        """Open a directory selection dialog."""
        # Release grab and topmost to allow dialog to work properly
        self.parent.grab_release()
        self.parent.attributes("-topmost", False)

        new_dir: str = filedialog.askdirectory()

        # Restore focus after dialog closes
        self.parent.lift()
        self.parent.focus_force()

        if new_dir:
            self.parent.dir = new_dir
            self.load_directory()

    def load_directory(self) -> None:
        """Load image file paths from the selected directory."""
        if not self.parent.dir:
            return

        dir_path = Path(self.parent.dir)
        self.parent.folder_name_label.configure(text=str(dir_path))  # Show full path instead of just name

        include_subdirs = self.parent.config_ui_data["include_subdirectories"]

        # For huge directories, show immediate feedback that loading is happening
        if include_subdirs:
            logger.info(f"Scanning directory {dir_path} (including subdirectories)")
        else:
            logger.info(f"Scanning directory {dir_path}")

        # Use scan_for_supported_images for all platforms - no need for special cases
        self.parent.image_rel_paths = scan_for_supported_images(
            str(dir_path),
            include_subdirs,
            self.parent.image_handler.is_supported_image,
        )

        # Initialize filtered paths and apply filters
        if hasattr(self.parent, "_apply_filters"):
            self.parent._apply_filters()
        else:
            self.parent.filtered_image_paths = self.parent.image_rel_paths.copy()
            self.parent._update_file_list()

        if self.parent.filtered_image_paths:
            # Find the index in the original list
            idx = self.parent.image_rel_paths.index(self.parent.filtered_image_paths[0])
            self.parent.navigation_manager.switch_to_image(idx)
        else:
            self.parent.clear_ui()

        # Regain focus after directory is loaded
        self.parent.lift()

        # Temporarily set topmost to ensure focus, then remove it
        self.parent.attributes("-topmost", True)
        self.parent.after(
            100, lambda: self.parent.attributes("-topmost", False)
        )



class CaptionManager:
    def __init__(self, parent: CaptionUI) -> None:
        self.parent: CaptionUI = parent

    def refresh_caption(self) -> None:
        """Refresh the caption entry with the current caption line."""
        self.parent.caption_entry.delete(0, "end")
        self.parent.caption_entry.insert(
            0, self.parent.caption_lines[self.parent.current_caption_line]
        )

    def next_caption_line(self, event: tk.Event | None = None) -> str:
        """Switch to the next caption line."""
        current_text: str = self.parent.caption_entry.get()
        self.parent.caption_lines[self.parent.current_caption_line] = (
            current_text
        )
        self.parent.current_caption_line = (
            self.parent.current_caption_line + 1
        ) % len(self.parent.caption_lines)
        self.parent.caption_line_var.set(
            self.parent.caption_line_values[
                self.parent.current_caption_line
            ]
        )
        self.refresh_caption()
        return "break"

    def on_caption_line_changed(self, value: str) -> None:
        """Handle changes in the selected caption line."""
        current_text: str = self.parent.caption_entry.get()
        self.parent.caption_lines[self.parent.current_caption_line] = (
            current_text
        )
        try:
            line_number: int = int(value.split(" ")[1]) - 1
            self.parent.current_caption_line = line_number
        except (ValueError, IndexError):
            self.parent.current_caption_line = 0
        self.refresh_caption()


class EditMode(Enum):
    DRAW = "draw"
    FILL = "fill"
