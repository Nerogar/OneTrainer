"""
OneTrainer Dataset Tools

This module defines a CTkToplevel window for image editing with the capbility to automatically
or manually caption and mask images from a loaded directory.
"""

import contextlib
import platform
import subprocess
import tkinter as tk
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from tkinter import filedialog
from typing import TypeAlias

from modules.module.Blip2Model import Blip2Model
from modules.module.BlipModel import BlipModel
from modules.module.ClipSegModel import ClipSegModel
from modules.module.MaskByColor import MaskByColor
from modules.module.RembgHumanModel import RembgHumanModel
from modules.module.RembgModel import RembgModel
from modules.module.WDModel import WDModel
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
from customtkinter import ThemeManager
from PIL import Image, ImageDraw, ImageTk

RGBColor: TypeAlias = tuple[int, int, int]
ImageCoordinates: TypeAlias = tuple[int, int, int, int]



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
    return sorted(results)


class CaptionUI(ctk.CTkToplevel):
    WINDOW_WIDTH: int = 1018
    WINDOW_HEIGHT: int = 768
    FILE_LIST_WIDTH: int = 250
    MASK_MIN_OPACITY: float = 0.3
    DEFAULT_BRUSH_SIZE: float = 0.01

    def __init__(
        self,
        parent: ctk.CTk,
        initial_dir: str | None = None,
        include_subdirectories: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(parent, *args, **kwargs)
        self.dir: str | None = initial_dir
        self.config_ui_data: dict = {
            "include_subdirectories": include_subdirectories
        }
        self.config_ui_state: UIState = UIState(self, self.config_ui_data)

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

        # Instantiate managers and handlers.
        self.mask_editor = MaskEditor(self)
        self.model_manager = ModelManager()
        self.navigation_manager = NavigationManager(self)
        self.caption_manager = CaptionManager(self)
        self.file_manager = FileManager(self)
        self.image_handler = ImageHandler(self)

        self._setup_window()
        self._create_layout()

        if initial_dir:
            self.file_manager.load_directory()

    def _setup_window(self) -> None:
        """Set up window properties."""
        self.title("Dataset Tools")
        self.geometry(f"{self.WINDOW_WIDTH}x{self.WINDOW_HEIGHT}")
        self.resizable(True, True)
        self.wait_visibility()
        self.focus_set()
        self.lift()  # Raise window to the top
        self.grab_set()  # Make this window modal

        # Schedule releasing topmost attribute after window is fully shown
        self.attributes("-topmost", True)
        self.after(100, lambda: self.attributes("-topmost", False))

        self.help_text: str = (
                "Keyboard shortcuts:\n\n"
                "Left/Right arrows: Navigate between images\n"
                "Tab: Switch between caption lines\n"
                "Return or Ctrl+S: Save changes\n"
                "Ctrl+M: Toggle mask display\n"
                "Ctrl+D: Switch to draw mode\n"
                "Ctrl+F: Switch to fill mode\n"
                "Ctrl+Z: Undo mask edit\n"
                "Ctrl+Y: Redo mask edit\n"
                "[ or ]: Decrease/increase brush size\n\n"
                "When editing masks:\n"
                "Left click: Add to mask\n"
                "Right click: Remove from mask\n"
                "Mouse wheel: Adjust brush size"
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
        if platform.system() == "Windows":
            self._create_icon_button(
                top_frame,
                0,
                3,
                "Explorer",
                self.file_manager.open_in_explorer,
                "explorer",
                "Open in File Explorer",
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
        top_frame.grid_columnconfigure(5, weight=1)
        self._create_icon_button(
            top_frame,
            0,
            6,
            "Help",
            self._show_help,
            "help",
            self.help_text,
        )

    def _create_main_content(self) -> None:
        """Create the main content area."""
        main_frame = ctk.CTkFrame(self, fg_color="#242424")
        main_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=10)
        main_frame.grid_columnconfigure(0, weight=0)
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)
        self._create_file_list(main_frame)
        self._create_editor_panel(main_frame)

    def _create_file_list(self, parent: ctk.CTkFrame) -> None:
        """Create the file list pane."""
        file_area_frame = ctk.CTkFrame(parent, fg_color="transparent")
        file_area_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        file_area_frame.grid_columnconfigure(0, weight=1)
        file_area_frame.grid_rowconfigure(0, weight=1)
        self.file_list = ctk.CTkScrollableFrame(
            file_area_frame,
            width=self.FILE_LIST_WIDTH,
            scrollbar_fg_color="transparent",
            scrollbar_button_hover_color="grey75",
            scrollbar_button_color="grey50",
        )
        self.file_list.grid(row=0, column=0, sticky="nsew")
        self._file_list_scrollbar = self.file_list._scrollbar
        self._file_list_scrollbar.grid_remove()

        header_frame = ctk.CTkFrame(self.file_list)
        header_frame.grid(
            row=0, column=0, sticky="ew", padx=2, pady=(2, 4)
        )
        header_frame.grid_columnconfigure(1, weight=1)
        folder_icon = load_icon("folder-tree", (20, 20))
        ctk.CTkLabel(header_frame, text="", image=folder_icon).grid(
            row=0, column=0, padx=(5, 3), pady=5
        )
        self.folder_name_label = ctk.CTkLabel(
            header_frame,
            text="No folder selected",
            anchor="w",
            font=("Segoe UI", 12, "bold"),
        )
        self.folder_name_label.grid(
            row=0, column=1, sticky="ew", padx=0, pady=5
        )
        self.image_labels = []

    def _create_editor_panel(self, parent: ctk.CTkFrame) -> None:
        """Create the editor panel with tools, image container, and caption area."""
        editor_frame = ctk.CTkFrame(parent, fg_color="#242424")
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
        tools_frame = ctk.CTkFrame(parent, fg_color="#333333")
        tools_frame.grid(row=0, column=0, sticky="new")
        for i in range(9):
            tools_frame.grid_columnconfigure(i, weight=0)
        tools_frame.grid_columnconfigure(5, weight=1)
        icons = {
            "draw": "draw",
            "fill": "fill",
            "save": "save",
            "undo": "undo",
            "redo": "redo",
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
            text="Enable Editing",
            variable=self.enable_mask_editing_var,
        ).grid(row=0, column=2, padx=5, pady=2)
        self.brush_opacity_entry = ctk.CTkEntry(tools_frame, width=40)
        self.brush_opacity_entry.insert(0, "1.0")
        self.brush_opacity_entry.grid(row=0, column=3, padx=5, pady=2)
        ctk.CTkLabel(tools_frame, text="Brush Opacity").grid(
            row=0, column=4, padx=2, pady=2
        )
        self._create_icon_button(
            tools_frame,
            0,
            6,
            "",
            self.file_manager.save_changes,
            icons["save"],
            "Save changes",
        )
        self._create_icon_button(
            tools_frame,
            0,
            7,
            "",
            lambda: self.mask_editor.undo_mask_edit(),
            icons["undo"],
            "Undo last edit",
        )
        self._create_icon_button(
            tools_frame,
            0,
            8,
            "",
            lambda: self.mask_editor.redo_mask_edit(),
            icons["redo"],
            "Redo last undone edit",
        )

    def _create_image_container(self, parent: ctk.CTkFrame) -> None:
        """Create the image display container."""
        self.image_container = ctk.CTkFrame(parent, fg_color="#242424")
        self.image_container.grid(row=1, column=0, sticky="nsew")
        self.image_container.grid_columnconfigure(0, weight=1)
        self.image_container.grid_rowconfigure(0, weight=1)
        default_width: int = 800
        default_height: int = 600
        self.image_container.grid_propagate(False)
        placeholder: Image.Image = Image.new(
            "RGB", (default_width, default_height), (32, 32, 32)
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
        self.bind("<Configure>", self.image_handler.on_resize)
        bind_mousewheel(
            self.image_label,
            {self.image_label.children["!label"]},
            self.mask_editor.adjust_brush_size,
        )

    def _create_caption_area(self, parent: ctk.CTkFrame) -> None:
        """Create the caption area for editing."""
        caption_frame = ctk.CTkFrame(parent, fg_color="#333333")
        caption_frame.grid(row=2, column=0, sticky="sew")
        caption_frame.grid_columnconfigure(0, weight=0)
        caption_frame.grid_columnconfigure(1, weight=1)
        self.caption_line_values: list[str] = [
            f"Line {i}" for i in range(1, 6)
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
        self._bind_key_events(self.caption_entry)

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
        self.bind("<bracketleft>", self.mask_editor.decrease_brush_size)
        self.bind("<bracketright>", self.mask_editor.increase_brush_size)
        self.bind("<Control-z>", self.mask_editor.undo_mask_edit)
        self.bind("<Control-y>", self.mask_editor.redo_mask_edit)

    def _update_file_list(self) -> None:
        """Refresh the file list UI with optimized updates."""
        # Store current scroll position to restore after update - use contextlib.suppress pattern
        current_scroll = 0  # Default value
        with contextlib.suppress(Exception):
            current_scroll = self.file_list._parent_canvas.yview()[0]

        # Clear existing widgets
        for widget in self.file_list.winfo_children():
            widget.destroy()

        self.image_labels = []

        # Create header with consistent Path usage
        header_frame = ctk.CTkFrame(self.file_list)
        header_frame.grid(
            row=0, column=0, sticky="ew", padx=2, pady=(2, 4)
        )
        header_frame.grid_columnconfigure(1, weight=1)
        folder_icon = load_icon("folder-tree", (20, 20))
        ctk.CTkLabel(header_frame, text="", image=folder_icon).grid(
            row=0, column=0, padx=(5, 3), pady=5
        )

        # Use pathlib consistently
        folder_path: str = (
            str(Path(self.dir).absolute())
            if self.dir
            else "No folder selected"
        )

        self.folder_name_label = ctk.CTkLabel(
            header_frame,
            text=folder_path,
            anchor="w",
            font=("Segoe UI", 12, "bold"),
            wraplength=self.FILE_LIST_WIDTH - 35,
        )
        self.folder_name_label.grid(
            row=0, column=1, sticky="ew", padx=0, pady=5
        )

        # Add virtualization for large directories
        MAX_VISIBLE_FILES = 500
        if len(self.image_rel_paths) > MAX_VISIBLE_FILES:
            warning = ctk.CTkLabel(
                self.file_list,
                text=f"Showing {MAX_VISIBLE_FILES} of {len(self.image_rel_paths)} files",
                text_color="orange",
                font=("Segoe UI", 10, "italic"),
            )
            warning.grid(row=1, column=0, sticky="ew", padx=2, pady=5)
            display_files = self.image_rel_paths[:MAX_VISIBLE_FILES]
            start_row = 2
        else:
            display_files = self.image_rel_paths
            start_row = 1

        # Create file labels
        for i, filename in enumerate(display_files):
            safe_filename: str = (
                filename
                if isinstance(filename, str)
                else filename.decode("utf-8", errors="replace")
            )

            label = ctk.CTkLabel(
                self.file_list,
                text=safe_filename,
                wraplength=self.FILE_LIST_WIDTH - 20,
                font=("Segoe UI", 11),
            )
            label.bind(
                "<Button-1>",
                lambda e, idx=i: self.navigation_manager.switch_to_image(
                    idx
                ),
            )
            label.grid(
                row=i + start_row,
                column=0,
                sticky="w",
                padx=2,
                pady=(1 if i == 0 else 2, 2),
            )
            self.image_labels.append(label)

        # Update scroll position and scrollbar
        self.after(
            100, lambda: self._restore_scroll_position(current_scroll)
        )
        self.after(200, self._update_scrollbar_visibility)

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
            print(f"Note: Could not adjust scrollbar visibility: {e}")

    def refresh_ui(self) -> None:
        """Refresh the image and caption UI."""
        if self.pil_image:
            self.image_handler.refresh_image()
            self.caption_manager.refresh_caption()

    def clear_ui(self) -> None:
        """Clear the current image, mask, and caption data."""
        if self.image_container and self.image_container.winfo_ismapped():
            width = max(10, self.image_container.winfo_width())
            height = max(10, self.image_container.winfo_height())
        else:
            width, height = 800, 600
        empty_image: Image.Image = Image.new(
            "RGB", (width, height), (32, 32, 32)
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
        """Open the auto-caption generation window."""
        if self.dir:
            dialog = GenerateCaptionsWindow(
                self,
                self.dir,
                self.config_ui_data["include_subdirectories"],
            )
            self.wait_window(dialog)
            if 0 <= self.current_image_index < len(self.image_rel_paths):
                self.navigation_manager.switch_to_image(
                    self.current_image_index
                )

    def _open_mask_window(self) -> None:
        """Open the auto-mask generation window."""
        if self.dir:
            dialog = GenerateMasksWindow(
                self,
                self.dir,
                self.config_ui_data["include_subdirectories"],
            )
            self.wait_window(dialog)
            if 0 <= self.current_image_index < len(self.image_rel_paths):
                self.navigation_manager.switch_to_image(
                    self.current_image_index
                )

    def _show_help(self) -> None:
        """Show help text (currently printed to console)."""
        print(self.help_text)


class ImageHandler:
    def __init__(self, parent: CaptionUI) -> None:
        self.parent: CaptionUI = parent

    def _calculate_display_dimensions(self) -> tuple[int, int, int, int]:
        """
        Calculate display dimensions to fit the image in the container while preserving aspect ratio.
        Returns (display_width, display_height, left_offset, top_offset).
        """
        container_width: int = max(
            10, self.parent.image_container.winfo_width()
        )
        container_height: int = max(
            10, self.parent.image_container.winfo_height()
        )
        if container_width <= 10:
            container_width = 800
        if container_height <= 10:
            container_height = 600

        image_width: int = self.parent.image_width
        image_height: int = self.parent.image_height
        if image_width <= 0 or image_height <= 0:
            return 0, 0, 0, 0

        padding: int = 20
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
        display_width, display_height, left_offset, top_offset = (
            self._calculate_display_dimensions()
        )
        self.parent.display_width = display_width
        self.parent.display_height = display_height
        self.parent.left_offset = left_offset
        self.parent.top_offset = top_offset
        container_width: int = self.parent.image_container.winfo_width()
        container_height: int = self.parent.image_container.winfo_height()
        canvas: Image.Image = Image.new(
            "RGB", (container_width, container_height), (32, 32, 32)
        )
        if display_width and display_height:
            resized_image: Image.Image = self.parent.pil_image.resize(
                (display_width, display_height), Image.Resampling.LANCZOS
            )
        else:
            resized_image = self.parent.pil_image.copy()
        if self.parent.pil_mask:
            resized_mask: Image.Image = self.parent.pil_mask.resize(
                (display_width, display_height), Image.Resampling.NEAREST
            )
            if self.parent.mask_editor.display_only_mask:
                final_image: Image.Image = resized_mask
            else:
                np_image: np.ndarray = (
                    np.array(resized_image, dtype=np.float32) / 255.0
                )
                np_mask: np.ndarray = (
                    np.array(resized_mask, dtype=np.float32) / 255.0
                )
                if np.min(np_mask) == 0:
                    np_mask = (
                        np_mask * (1.0 - self.parent.MASK_MIN_OPACITY)
                        + self.parent.MASK_MIN_OPACITY
                    )
                elif np.min(np_mask) < 1:
                    min_val: float = float(np.min(np_mask))
                    np_mask = (np_mask - min_val) / (1.0 - min_val) * (
                        1.0 - self.parent.MASK_MIN_OPACITY
                    ) + self.parent.MASK_MIN_OPACITY
                np_result: np.ndarray = (
                    np_image * np_mask * 255.0
                ).astype(np.uint8)
                final_image = Image.fromarray(np_result, mode="RGB")
        else:
            final_image = resized_image
        canvas.paste(final_image, (left_offset, top_offset))
        self.parent.image_tk = ImageTk.PhotoImage(canvas)
        self.parent.image_label.configure(image=self.parent.image_tk)

    def update_image_container_size(self) -> None:
        """Update container size and refresh image display."""
        if (
            self.parent.image_container.winfo_width() > 10
            and self.parent.image_container.winfo_height() > 10
        ):
            self.refresh_image()
        else:
            self.parent.image_container.config(width=800, height=600)
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
            print(f"Error loading image {image_path}: {e}")
            self.parent.pil_image = None

        mask_path = image_path.with_name(f"{image_path.stem}-masklabel.png")
        if mask_path.exists():
            try:
                self.parent.pil_mask = Image.open(mask_path).convert("RGB")
                self.parent.mask_editor.reset_for_new_image()
            except Exception as e:
                print(f"Error loading mask {mask_path}: {e}")
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
                print(f"Error loading caption {caption_path}: {e}")
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


class NavigationManager:
    def __init__(self, parent: CaptionUI) -> None:
        self.parent: CaptionUI = parent

    def switch_to_image(self, index: int) -> None:
        """Switch to the image at the given index."""
        if (
            0
            <= self.parent.current_image_index
            < len(self.parent.image_labels)
        ):
            self.parent.image_labels[
                self.parent.current_image_index
            ].configure(
                text_color=ThemeManager.theme["CTkLabel"]["text_color"],
                fg_color="transparent",
                corner_radius=0,
            )
        self.parent.current_image_index = index
        if 0 <= index < len(self.parent.image_labels):
            self.parent.image_labels[index].configure(
                text_color="#5AD9ED",
                fg_color="#454545",
                corner_radius=6,
            )
            self.parent.image_handler.load_image_data()
            self.parent.refresh_ui()
        else:
            self.parent.clear_ui()

    def next_image(self, event: tk.Event | None = None) -> str:
        """Switch to the next image."""
        if self.parent.image_rel_paths and (
            self.parent.current_image_index + 1
        ) < len(self.parent.image_rel_paths):
            self.parent.file_manager.save_changes()
            self.switch_to_image(self.parent.current_image_index + 1)
        return "break"

    def previous_image(self, event: tk.Event | None = None) -> str:
        """Switch to the previous image."""
        if (
            self.parent.image_rel_paths
            and (self.parent.current_image_index - 1) >= 0
        ):
            self.parent.file_manager.save_changes()
            self.switch_to_image(self.parent.current_image_index - 1)
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
            "Blip": BlipModel,
            "Blip2": Blip2Model,
            "WD14 VIT v2": WDModel,
        }

        self._masking_registry = {
            "ClipSeg": ClipSegModel,
            "Rembg": RembgModel,
            "Rembg-Human": RembgHumanModel,
            "Hex Color": MaskByColor,
        }

        self.masking_model = None
        self.captioning_model = None

    def get_available_captioning_models(self) -> list[str]:
        """Return a list of available captioning models."""
        return list(self._captioning_registry.keys())

    def get_available_masking_models(self) -> list[str]:
        """Return a list of available masking models."""
        return list(self._masking_registry.keys())

    def load_masking_model(
        self, model: str
    ) -> ClipSegModel | RembgModel | RembgHumanModel | MaskByColor | None:
        """Load the specified masking model, unloading any captioning model."""
        self.captioning_model = None

        if model not in self._masking_registry:
            print(f"Unknown masking model: {model}")
            return None

        model_class = self._masking_registry[model]

        if self.masking_model is None or not isinstance(
            self.masking_model, model_class
        ):
            print(f"Loading {model} model, this may take a while")
            self.masking_model = model_class(self.device, torch.float32)

        return self.masking_model

    def load_captioning_model(
        self, model: str
    ) -> BlipModel | Blip2Model | WDModel | None:
        """Load the specified captioning model, unloading any masking model."""
        self.masking_model = None

        if model not in self._captioning_registry:
            print(f"Unknown captioning model: {model}")
            return None

        model_class = self._captioning_registry[model]

        if self.captioning_model is None or not isinstance(
            self.captioning_model, model_class
        ):
            print(f"Loading {model} model, this may take a while")
            self.captioning_model = model_class(
                self.device, self.precision
            )

        return self.captioning_model

    def get_masking_model(
        self,
    ) -> ClipSegModel | RembgModel | RembgHumanModel | MaskByColor | None:
        return self.masking_model

    def get_captioning_model(
        self,
    ) -> BlipModel | Blip2Model | WDModel | None:
        return self.captioning_model


class FileManager:
    def __init__(self, parent: CaptionUI) -> None:
        self.parent: CaptionUI = parent
        self._last_saved_caption = {}  # Simple cache to track saved content
        self._mask_modified = {}  # Track if mask has been modified

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

        # Handle mask saving if it exists and was modified
        if self.parent.pil_mask:
            mask_path = image_path.with_name(
                f"{image_path.stem}-masklabel.png"
            )
            if self._mask_modified.get(image_key, False):
                try:
                    self.parent.pil_mask.save(mask_path)
                    print(f"Saved mask to {mask_path}")
                    self._mask_modified[image_key] = False
                except Exception as e:
                    print(f"Error saving mask: {e}")

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
                    print(f"Saved caption to {caption_path}")
                    self._last_saved_caption[image_key] = caption_content
                except Exception as e:
                    print(f"Error saving caption: {e}")

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
            print(f"Error opening file in explorer: {e}")

    def open_directory(self) -> None:
        """Open a directory selection dialog."""
        new_dir: str = filedialog.askdirectory()
        if new_dir:
            self.parent.dir = new_dir
            self.load_directory()

    def load_directory(self) -> None:
        """Load image file paths from the selected directory."""
        if not self.parent.dir:
            return

        dir_path = Path(self.parent.dir)
        self.parent.folder_name_label.configure(text=dir_path.name)

        include_subdirs = self.parent.config_ui_data[
            "include_subdirectories"
        ]

        # For huge directories, show immediate feedback that loading is happening
        if include_subdirs:
            print(
                f"Scanning directory {dir_path} (including subdirectories)"
            )
        else:
            print(f"Scanning directory {dir_path}")

        # Use pathlib for scanning directories
        if not include_subdirs and platform.system() == "Windows":
            # Fast path for non-recursive Windows scanning
            self.parent.image_rel_paths = []
            is_supported = self.parent.image_handler.is_supported_image
            for entry in dir_path.iterdir():
                if entry.is_file() and is_supported(entry.name):
                    self.parent.image_rel_paths.append(entry.name)
            self.parent.image_rel_paths.sort()
        else:
            # Fall back to the existing scan function for recursive or non-Windows
            self.parent.image_rel_paths = scan_for_supported_images(
                str(dir_path),
                include_subdirs,
                self.parent.image_handler.is_supported_image,
            )

        self.parent._update_file_list()
        if self.parent.image_rel_paths:
            self.parent.navigation_manager.switch_to_image(0)
        else:
            self.parent.clear_ui()

        # Regain focus after directory is loaded
        self.parent.focus_set()
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


class MaskEditor:
    DEFAULT_BRUSH_SIZE: float = 0.02

    def __init__(self, parent: CaptionUI) -> None:
        self.parent: CaptionUI = parent
        self.mask_draw_x: float = 0.0
        self.mask_draw_y: float = 0.0
        self.mask_draw_radius: float = self.DEFAULT_BRUSH_SIZE
        self.mask_editing_mode: EditMode = EditMode.DRAW
        self.display_only_mask: bool = False
        self.mask_history: list[Image.Image] = []
        self.mask_history_position: int = -1
        self.mask_history_limit: int = 30
        self.is_editing: bool = False
        self.edit_started: bool = False
        self._cached_image_dimensions: tuple[int, int] = (0, 0)

    def decrease_brush_size(self, event: tk.Event | None = None) -> str:
        """Decrease the brush size."""
        self.mask_draw_radius = max(0.0025, self.mask_draw_radius / 1.25)
        return "break"

    def increase_brush_size(self, event: tk.Event | None = None) -> str:
        """Increase the brush size."""
        self.mask_draw_radius = min(0.5, self.mask_draw_radius * 1.25)
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
        start_x, start_y, end_x, end_y = self._convert_screen_to_mask_coordinates(event)
        if start_x == end_x == 0 and start_y == end_y == 0:
            return
        is_left: bool = bool(event.state & 0x0100 or event.num == 1)
        is_right: bool = bool(event.state & 0x0400 or event.num == 3)
        if not (is_left or is_right) and not self.is_editing:
            return
        if self.mask_editing_mode == EditMode.DRAW:
            self._draw_mask(
                start_x, start_y, end_x, end_y, is_left, is_right
            )
        elif self.mask_editing_mode == EditMode.FILL:
            self._fill_mask(start_x, start_y, is_left, is_right)

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
        # Remove the division by the scaling factor (assume event.x/y are already in widget coordinates)
        left_offset: int = self.parent.left_offset
        top_offset: int = self.parent.top_offset
        display_width: int = self.parent.display_width
        display_height: int = self.parent.display_height
        image_width: int = self.parent.image_width
        image_height: int = self.parent.image_height
        # Use event.x and event.y directly
        event_x: float = event.x
        event_y: float = event.y
        image_x: float = event_x - left_offset
        image_y: float = event_y - top_offset
        if not (
            0 <= image_x < display_width and 0 <= image_y < display_height
        ):
            return 0, 0, 0, 0
        start_x: int = int(image_x * image_width / display_width)
        start_y: int = int(image_y * image_height / display_height)
        # Use previous stored coordinates for a smooth line (if available)
        if hasattr(self, "mask_draw_x") and hasattr(self, "mask_draw_y"):
            prev_image_x: float = self.mask_draw_x - left_offset
            prev_image_y: float = self.mask_draw_y - top_offset
            if (
                0 <= prev_image_x < display_width
                and 0 <= prev_image_y < display_height
            ):
                end_x: int = int(
                    prev_image_x * image_width / display_width
                )
                end_y: int = int(
                    prev_image_y * image_height / display_height
                )
            else:
                end_x, end_y = start_x, start_y
        else:
            end_x, end_y = start_x, start_y
        # Store current event positions for the next call.
        self.mask_draw_x = event.x
        self.mask_draw_y = event.y
        return start_x, start_y, end_x, end_y


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
        color: RGBColor | None = self._determine_brush_mask_color(is_left)
        if color:
            self._ensure_mask_exists(is_left)
            if not self.edit_started:
                self._save_mask_to_history()
                self.edit_started = True
            max_dimension: int = max(
                self.parent.pil_mask.width, self.parent.pil_mask.height
            )
            radius: int = int(self.mask_draw_radius * max_dimension)
            if radius <= 0 or (start_x == end_x == start_y == end_y == 0):
                return
            draw: ImageDraw.ImageDraw = ImageDraw.Draw(
                self.parent.pil_mask
            )
            line_width: int = 2 * radius + 1
            draw.line(
                (start_x, start_y, end_x, end_y),
                fill=color,
                width=line_width,
            )
            draw.ellipse(
                (
                    start_x - radius,
                    start_y - radius,
                    start_x + radius,
                    start_y + radius,
                ),
                fill=color,
            )
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
            self.parent.image_handler.refresh_image()

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
        """Adjust the brush size based on mouse wheel movement."""
        multiplier: float = 1.0 + (
            delta * (0.03 if self.mask_draw_radius < 0.05 else 0.05)
        )
        self.mask_draw_radius = max(
            0.0025, min(0.5, self.mask_draw_radius * multiplier)
        )

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

    def switch_to_brush_mode(
        self, event: tk.Event | None = None
    ) -> str | None:
        """Switch to draw mask mode."""
        self.mask_editing_mode = EditMode.DRAW
        return "break" if event else None

    def switch_to_fill_mode(
        self, event: tk.Event | None = None
    ) -> str | None:
        """Switch to fill mask mode."""
        self.mask_editing_mode = EditMode.FILL
        return "break" if event else None

    def toggle_mask_visibility_mode(self, event: tk.Event | None = None) -> str:
        """Toggle between displaying only the mask or the combined image."""
        self.display_only_mask = not self.display_only_mask
        self.parent.image_handler.refresh_image()
        return "break"
