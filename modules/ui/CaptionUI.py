import os
import platform
import subprocess
from tkinter import filedialog

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
from customtkinter import ScalingTracker, ThemeManager
from PIL import Image, ImageDraw


class CaptionUI(ctk.CTkToplevel):
    """Multi-purpose image editor with captioning and masking functionality."""

    # UI Constants
    WINDOW_WIDTH = 1018
    WINDOW_HEIGHT = 768
    IMAGE_CONTAINER_WIDTH = 768
    IMAGE_CONTAINER_HEIGHT = 512
    FILE_LIST_WIDTH = 250
    MASK_MIN_OPACITY = 0.3
    DEFAULT_BRUSH_SIZE = 0.01

    def __init__(
        self,
        parent,
        initial_dir: str | None = None,
        include_subdirectories: bool = False,
        *args,
        **kwargs,
    ):
        """Initialize the Image Editor UI."""
        super().__init__(parent, *args, **kwargs)

        # Initialize basic properties
        self.dir = initial_dir
        self.config_ui_data = {
            "include_subdirectories": include_subdirectories
        }
        self.config_ui_state = UIState(self, self.config_ui_data)

        # Image data and state variables
        self.image_rel_paths: list[str] = []
        self.current_image_index = -1
        self.pil_image: Image.Image | None = None
        self.pil_mask: Image.Image | None = None
        self.image_width = 0
        self.image_height = 0

        # Mask editing state
        self.mask_draw_x = 0
        self.mask_draw_y = 0
        self.mask_draw_radius = self.DEFAULT_BRUSH_SIZE
        self.mask_editing_mode = "draw"
        self.display_only_mask = False

        # Mask undo/redo
        self.mask_history = []
        self.mask_history_position = -1
        self.mask_history_limit = 6
        self.is_editing = False
        self.edit_started = False

        # Caption state
        self.caption_lines: list[str] = []
        self.current_caption_line = 0

        # Models
        self.masking_model = None
        self.captioning_model = None

        # Setup UI
        self._setup_window()
        self._create_layout()

        # Load directory if provided
        if initial_dir:
            self.load_directory()

    def _setup_window(self) -> None:
        """Configure the main window properties."""
        self.title("OneTrainer Image Editor")
        self.geometry(f"{self.WINDOW_WIDTH}x{self.WINDOW_HEIGHT}")
        self.resizable(True, True)
        self.wait_visibility()
        self.focus_set()

        self.help_text = """
Keyboard shortcuts:
Up/Down arrows: Navigate between images
Tab: Switch between caption lines
Return: Save changes
Ctrl+M: Toggle mask display
Ctrl+D: Switch to draw mode
Ctrl+F: Switch to fill mode

When editing masks:
Left click: Add to mask
Right click: Remove from mask
Mouse wheel: Adjust brush size
        """

    def _create_layout(self) -> None:
        """Create the main UI layout."""
        # Configure grid layout
        self.grid_rowconfigure(0, weight=0)  # Top bar
        self.grid_rowconfigure(1, weight=1)  # Content area
        self.grid_columnconfigure(0, weight=1)

        # Create UI variables
        self.enable_mask_editing_var = ctk.BooleanVar(value=False)

        # Create top bar
        self._create_top_bar()

        # Create main content area
        self._create_main_content()

    def _create_top_bar(self) -> None:
        """Create the top action bar."""
        # Use the correct theme color from customtkinter for consistency
        top_frame = ctk.CTkFrame(self, corner_radius=0)
        top_frame.grid(row=0, column=0, sticky="new", padx=0, pady=0)

        # Load icons
        icon_size = (24, 24)
        icons = {
            "load": load_icon("load", icon_size),
            "auto-mask": load_icon("auto-mask", icon_size),
            "auto-caption": load_icon("auto-caption", icon_size),
            "explorer": load_icon("explorer", icon_size),
            "help": load_icon("help", icon_size),
        }

        # Create action buttons
        components.icon_button(
            top_frame,
            0,
            0,
            "Load",
            self._open_directory,
            image=icons["load"],
            tooltip="Load a directory of images",
        )
        components.icon_button(
            top_frame,
            0,
            1,
            "Auto-Mask",
            self._open_mask_window,
            image=icons["auto-mask"],
            tooltip="Generate masks automatically",
        )
        components.icon_button(
            top_frame,
            0,
            2,
            "Auto-Caption",
            self._open_caption_window,
            image=icons["auto-caption"],
            tooltip="Generate captions automatically",
        )

        # Explorer button for Windows
        if platform.system() == "Windows":
            components.icon_button(
                top_frame,
                0,
                3,
                "Explorer",
                self._open_in_explorer,
                image=icons["explorer"],
                tooltip="Open in File Explorer",
            )

        # Include subdirectories switch
        components.switch(
            top_frame,
            0,
            4,
            self.config_ui_state,
            "include_subdirectories",
            text="Include Subdirs",
            tooltip="Include subdirectories when loading images",
        )

        # Expandable space
        top_frame.grid_columnconfigure(5, weight=1)

        # Help button
        components.icon_button(
            top_frame,
            0,
            6,
            "Help",
            self._show_help,
            image=icons["help"],
            tooltip=self.help_text,
        )

    def _create_main_content(self) -> None:
        """Create the main content area with file list and editor panel."""
        main_frame = ctk.CTkFrame(self, fg_color="#242424")
        main_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=10)

        # Configure grid layout
        main_frame.grid_columnconfigure(
            0, weight=0
        )  # File list (fixed width)
        main_frame.grid_columnconfigure(
            1, weight=1
        )  # Editor panel (expandable)
        main_frame.grid_rowconfigure(0, weight=1)

        # Create file list panel
        self._create_file_list(main_frame)

        # Create editor panel
        self._create_editor_panel(main_frame)

    def _create_file_list(self, parent) -> None:
        """Create the file list panel."""
        # Create a frame to contain the scrollable list
        file_area_frame = ctk.CTkFrame(parent, fg_color="transparent")
        file_area_frame.grid(
            row=0, column=0, sticky="nsew", padx=(0, 10), pady=0
        )
        file_area_frame.grid_columnconfigure(0, weight=1)
        file_area_frame.grid_rowconfigure(
            0, weight=1
        )  # Make file list expandable

        # Create scrollable frame for file list
        self.file_list = ctk.CTkScrollableFrame(
            file_area_frame,
            width=self.FILE_LIST_WIDTH,
            scrollbar_fg_color="transparent",
            scrollbar_button_hover_color="grey75",
            scrollbar_button_color="grey50",
        )
        self.file_list.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)

        # Store reference to the scrollbar for dynamic visibility management
        self._file_list_scrollbar = self.file_list._scrollbar

        # Hide the scrollbar initially - only show it when needed
        self._file_list_scrollbar.grid_remove()

        # Create header with folder icon and name inside the scrollable frame
        header_frame = ctk.CTkFrame(self.file_list)
        header_frame.grid(
            row=0,
            column=0,
            sticky="ew",
            padx=2,
            pady=(2, 4),  # Reduced bottom padding
        )
        header_frame.grid_columnconfigure(
            1, weight=1
        )  # Make folder name expand

        # Add folder icon
        folder_icon = load_icon("folder-tree", (20, 20))
        folder_icon_label = ctk.CTkLabel(
            header_frame, text="", image=folder_icon
        )
        folder_icon_label.grid(row=0, column=0, padx=(5, 3), pady=5)

        # Add folder name label (will be updated when directory is loaded)
        self.folder_name_label = ctk.CTkLabel(
            header_frame,
            text="No folder selected",
            anchor="w",
            font=("Segoe UI", 12, "bold"),
        )
        self.folder_name_label.grid(
            row=0, column=1, sticky="ew", padx=0, pady=5
        )

        self.image_labels = []  # Will be populated when loading directory

    def _create_editor_panel(self, parent) -> None:
        """Create the right-side editor panel."""
        editor_frame = ctk.CTkFrame(parent, fg_color="#242424")
        editor_frame.grid(row=0, column=1, sticky="nsew", padx=0, pady=0)

        # Configure grid layout
        editor_frame.grid_columnconfigure(0, weight=1)
        editor_frame.grid_rowconfigure(
            0, weight=0
        )  # Tools bar (fixed height)
        editor_frame.grid_rowconfigure(
            1, weight=1
        )  # Image container (expandable)
        editor_frame.grid_rowconfigure(
            2, weight=0
        )  # Caption area (fixed height)

        # Create tools bar - remove pady here to eliminate the gap
        self._create_tools_bar(editor_frame)

        # Create image container
        self._create_image_container(editor_frame)

        # Create caption area
        self._create_caption_area(editor_frame)

    def _create_tools_bar(self, parent) -> None:
        """Create the tools bar with mask editing controls."""
        # Use the same frame properties as parent for visual consistency
        tools_frame = ctk.CTkFrame(parent, fg_color="#333333")
        tools_frame.grid(row=0, column=0, sticky="new", padx=0, pady=0)

        # Configure grid layout
        tools_frame.grid_columnconfigure(0, weight=0)  # Draw button
        tools_frame.grid_columnconfigure(1, weight=0)  # Fill button
        tools_frame.grid_columnconfigure(
            2, weight=0
        )  # Enable editing checkbox
        tools_frame.grid_columnconfigure(3, weight=0)  # Opacity entry
        tools_frame.grid_columnconfigure(4, weight=0)  # Opacity label
        tools_frame.grid_columnconfigure(
            5, weight=1
        )  # Expandable space - moved to here
        tools_frame.grid_columnconfigure(6, weight=0)  # Save button
        tools_frame.grid_columnconfigure(7, weight=0)  # Undo button
        tools_frame.grid_columnconfigure(8, weight=0)  # Redo button

        # Load icons
        icons = {
            "draw": load_icon("draw", (24, 24)),
            "fill": load_icon("fill", (24, 24)),
            "save": load_icon("save", (24, 24)),
            "undo": load_icon("undo", (24, 24)),
            "redo": load_icon("redo", (24, 24)),
        }

        # Create mask editing controls
        components.icon_button(
            tools_frame,
            0,
            0,
            "Draw",
            self._draw_mask_mode,
            image=icons["draw"],
            tooltip="Draw mask with brush",
        )
        components.icon_button(
            tools_frame,
            0,
            1,
            "Fill",
            self._fill_mask_mode,
            image=icons["fill"],
            tooltip="Fill areas of the mask",
        )

        # Enable mask editing checkbox
        enable_editing_checkbox = ctk.CTkCheckBox(
            tools_frame,
            text="Enable Editing",
            variable=self.enable_mask_editing_var,
        )
        enable_editing_checkbox.grid(row=0, column=2, padx=5, pady=2)

        # Brush opacity control
        self.brush_opacity_entry = ctk.CTkEntry(tools_frame, width=40)
        self.brush_opacity_entry.insert(0, "1.0")
        self.brush_opacity_entry.grid(row=0, column=3, padx=5, pady=2)

        opacity_label = ctk.CTkLabel(tools_frame, text="Brush Opacity")
        opacity_label.grid(row=0, column=4, padx=2, pady=2)

        # Column 5 is now our expandable space, buttons come after

        # Add save, undo and redo buttons on the right side
        components.icon_button(
            tools_frame,
            0,
            6,
            "",
            self._save_changes,
            image=icons["save"],
            tooltip="Save changes",
        )

        components.icon_button(
            tools_frame,
            0,
            7,
            "",
            self._undo_mask_edit,
            image=icons["undo"],
            tooltip="Undo last edit",
        )

        components.icon_button(
            tools_frame,
            0,
            8,
            "",
            self._redo_mask_edit,
            image=icons["redo"],
            tooltip="Redo last undone edit",
        )

    def _create_image_container(self, parent) -> None:
        """Create the image display container."""
        self.image_container = ctk.CTkFrame(
            parent, fg_color="#242424"
        )
        self.image_container.grid(
            row=1, column=0, sticky="nsew", padx=0, pady=0
        )
        self.image_container.grid_columnconfigure(0, weight=1)
        self.image_container.grid_rowconfigure(0, weight=1)

        # Create image display with initial dimensions (will be updated after layout)
        self.image = ctk.CTkImage(
            light_image=Image.new(
                "RGB",
                (self.IMAGE_CONTAINER_WIDTH, self.IMAGE_CONTAINER_HEIGHT),
                (32, 32, 32),
            ),
            size=(self.IMAGE_CONTAINER_WIDTH, self.IMAGE_CONTAINER_HEIGHT),
        )

        self.image_label = ctk.CTkLabel(
            self.image_container, text="", image=self.image
        )
        self.image_label.grid(row=0, column=0, sticky="nsew")

        # Add this: Update image dimensions after initial layout
        self.after(100, self._update_image_container_size)

        # Bind mask editing events
        self.image_label.bind("<Motion>", self._handle_mask_edit)
        self.image_label.bind("<Button-1>", self._handle_mask_edit_start)
        self.image_label.bind(
            "<ButtonRelease-1>", self._handle_mask_edit_end
        )
        self.image_label.bind("<Button-3>", self._handle_mask_edit_start)
        self.image_label.bind(
            "<ButtonRelease-3>", self._handle_mask_edit_end
        )

        # Bind resize event
        self.bind("<Configure>", self._on_resize)

        # Bind mouse wheel for brush size adjustment
        bind_mousewheel(
            self.image_label,
            {self.image_label.children["!label"]},
            self._adjust_brush_size,
        )

    def _update_image_container_size(self):
        """Update the image to use actual container size once it's available."""
        container_width = self.image_container.winfo_width()
        container_height = self.image_container.winfo_height()

        if container_width > 10 and container_height > 10:
            # Instead of just updating the size, do a full refresh
            self._refresh_image()
        else:
            # Wait a bit longer and try again
            self.after(100, self._update_image_container_size)

    def _create_caption_area(self, parent) -> None:
        """Create the caption editing area."""
        # Use the correct theme color from customtkinter for consistency
        caption_frame = ctk.CTkFrame(parent, fg_color="#333333")
        caption_frame.grid(row=2, column=0, sticky="sew", padx=0, pady=0)

        # Configure grid layout
        caption_frame.grid_columnconfigure(0, weight=0)  # Line dropdown
        caption_frame.grid_columnconfigure(1, weight=1)  # Caption entry

        # Create caption line selector
        self.caption_line_values = [
            "Line 1",
            "Line 2",
            "Line 3",
            "Line 4",
            "Line 5",
        ]
        self.caption_line_var = ctk.StringVar(
            value=self.caption_line_values[0]
        )

        caption_line_dropdown = ctk.CTkOptionMenu(
            caption_frame,
            values=self.caption_line_values,
            variable=self.caption_line_var,
            command=self._on_caption_line_changed,
        )
        caption_line_dropdown.grid(row=0, column=0, padx=5, pady=5)

        # Create caption entry
        self.caption_entry = ctk.CTkEntry(caption_frame)
        self.caption_entry.grid(
            row=0, column=1, padx=5, pady=5, sticky="ew"
        )

        # Bind keyboard shortcuts
        self._bind_key_events(self.caption_entry)

    def _bind_key_events(self, component) -> None:
        """Bind keyboard shortcuts to the given component."""
        component.bind("<Down>", self._next_image)
        component.bind("<Up>", self._previous_image)
        component.bind("<Return>", self._save_changes)
        component.bind("<Tab>", self._next_caption_line)
        component.bind("<Control-m>", self._toggle_mask_display)
        component.bind("<Control-d>", self._draw_mask_mode)
        component.bind("<Control-f>", self._fill_mask_mode)

    # File handling and navigation methods
    def load_directory(self) -> None:
        """Load images from the current directory."""
        if not self.dir:
            return

        # Update folder name in header
        folder_name = os.path.basename(os.path.normpath(self.dir))
        self.folder_name_label.configure(text=folder_name)

        self._scan_directory()
        self._update_file_list()

        if self.image_rel_paths:
            self._switch_to_image(0)
        else:
            self._clear_ui()

    def _scan_directory(self) -> None:
        """Scan the directory for supported image files."""
        self.image_rel_paths = []

        if not self.dir or not os.path.isdir(self.dir):
            return

        include_subdirs = self.config_ui_data["include_subdirectories"]

        if include_subdirs:
            for root, _, files in os.walk(self.dir):
                for filename in files:
                    if self._is_supported_image(filename):
                        rel_path = os.path.relpath(
                            os.path.join(root, filename), self.dir
                        )
                        self.image_rel_paths.append(rel_path)
        else:
            for filename in os.listdir(self.dir):
                if os.path.isfile(
                    os.path.join(self.dir, filename)
                ) and self._is_supported_image(filename):
                    self.image_rel_paths.append(filename)

        # Sort for consistent order
        self.image_rel_paths.sort()

    def _update_file_list(self) -> None:
        """Update the file list with loaded images."""
        # Clear existing labels
        for widget in self.file_list.winfo_children():
            widget.destroy()

        self.image_labels = []

        # Recreate the header (unchanged)
        header_frame = ctk.CTkFrame(self.file_list)
        header_frame.grid(row=0, column=0, sticky="ew", padx=2, pady=(2, 4))
        header_frame.grid_columnconfigure(1, weight=1)
        folder_icon = load_icon("folder-tree", (20, 20))
        folder_icon_label = ctk.CTkLabel(header_frame, text="", image=folder_icon)
        folder_icon_label.grid(row=0, column=0, padx=(5, 3), pady=5)
        folder_path = os.path.abspath(self.dir) if self.dir else "No folder selected"
        self.folder_name_label = ctk.CTkLabel(
            header_frame,
            text=folder_path,
            anchor="w",
            font=("Segoe UI", 12, "bold"),
            wraplength=self.FILE_LIST_WIDTH - 35,
        )
        self.folder_name_label.grid(row=0, column=1, sticky="ew", padx=0, pady=5)

        # Create new labels for each file without per-iteration try/except.
        # Sanitize the filename (this ensures non-ASCII is handled correctly).
        for i, filename in enumerate(self.image_rel_paths):
            safe_filename = (
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
            label.bind("<Button-1>", lambda e, idx=i: self._switch_to_image(idx))
            top_padding = 1 if i == 0 else 2
            label.grid(
                row=i + 1, column=0, sticky="w", padx=2, pady=(top_padding, 2)
            )
            self.image_labels.append(label)

        # Update scrollbar after updating contents
        self.after(100, self._update_scrollbar_visibility)

    def _update_scrollbar_visibility(self) -> None:
        """Update scrollbar visibility based on content size."""
        try:
            # Check if customtkinter uses a different internal structure
            # For current versions, try accessing canvas through these methods
            if hasattr(self.file_list, "_scrollbar_frame") and hasattr(
                self.file_list, "_parent_canvas"
            ):
                # Newer customtkinter versions
                canvas = self.file_list._parent_canvas
                inner_frame = self.file_list._scrollbar_frame

                # Get size information
                canvas_height = canvas.winfo_height()
                frame_height = inner_frame.winfo_reqheight()

                # Show scrollbar only if content exceeds visible area
                if frame_height > canvas_height:
                    self._file_list_scrollbar.grid()
                else:
                    self._file_list_scrollbar.grid_remove()
            else:
                # For older or different versions, fall back to setting
                # a fixed visibility based on item count
                if len(self.image_rel_paths) > 15:  # Arbitrary threshold
                    self._file_list_scrollbar.grid()
                else:
                    self._file_list_scrollbar.grid_remove()

        except (AttributeError, Exception) as e:
            # If any error occurs, ensure scrollbar is always visible
            # This is safer than potentially hiding a needed scrollbar
            if hasattr(self, "_file_list_scrollbar"):
                self._file_list_scrollbar.grid()
            print(f"Note: Could not adjust scrollbar visibility: {e}")

    def _switch_to_image(self, index: int) -> None:
        """Switch to the image at the given index."""
        # Clear highlight from current image
        if 0 <= self.current_image_index < len(self.image_labels):
            self.image_labels[self.current_image_index].configure(
                text_color=ThemeManager.theme["CTkLabel"]["text_color"],
                fg_color="transparent",  # Reset background color
                corner_radius=0,  # Reset corner radius
            )

        # Update current index
        self.current_image_index = index

        if 0 <= index < len(self.image_labels):
            # Highlight selected image with blue text and gray rounded background
            self.image_labels[index].configure(
                text_color="#5AD9ED",  # Blue text instead of red
                fg_color="#454545",  # Dark gray background
                corner_radius=6,  # Rounded corners
            )

            # Load image data
            self._load_image_data()
            self._refresh_ui()
        else:
            self._clear_ui()

    def _load_image_data(self) -> None:
        """Load the current image, mask, and captions."""
        if not (0 <= self.current_image_index < len(self.image_rel_paths)):
            return

        image_path = os.path.join(
            self.dir, self.image_rel_paths[self.current_image_index]
        )

        # Load image
        try:
            self.pil_image = Image.open(image_path).convert("RGB")
            self.image_width = self.pil_image.width
            self.image_height = self.pil_image.height
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            self.pil_image = None

        # Load mask if exists
        mask_path = os.path.splitext(image_path)[0] + "-masklabel.png"
        if os.path.exists(mask_path):
            try:
                self.pil_mask = Image.open(mask_path).convert("RGB")

                # Reset history and save initial state
                self.mask_history = []
                self.mask_history_position = -1
                if self.pil_mask:
                    self.mask_history.append(self.pil_mask.copy())
                    self.mask_history_position = 0
            except Exception as e:
                print(f"Error loading mask {mask_path}: {e}")
                self.pil_mask = None
        else:
            self.pil_mask = None
            self.mask_history = []
            self.mask_history_position = -1

        # Load caption if exists
        caption_path = os.path.splitext(image_path)[0] + ".txt"
        if os.path.exists(caption_path):
            try:
                with open(caption_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    self.caption_lines = content.split("\n")
                    # Ensure at least 5 lines (pad with empty strings)
                    self.caption_lines.extend(
                        [""] * (5 - len(self.caption_lines))
                    )
                    # Limit to 5 lines
                    self.caption_lines = self.caption_lines[:5]
            except Exception as e:
                print(f"Error loading caption {caption_path}: {e}")
                self.caption_lines = [""] * 5
        else:
            self.caption_lines = [""] * 5

        # Reset current caption line
        self.current_caption_line = 0
        self.caption_line_var.set(self.caption_line_values[0])
        self.caption_entry.delete(0, "end")
        self.caption_entry.insert(0, self.caption_lines[0])

    def _refresh_ui(self) -> None:
        """Refresh the UI with current image data."""
        if not self.pil_image:
            return

        self._refresh_image()
        self._refresh_caption()

    def _on_resize(self, event=None) -> None:
        """Handle window resize events."""
        # Only refresh if we have an image and the container is visible
        if self.pil_image and self.image_container.winfo_ismapped():
            # Add a small delay to avoid excessive refreshes during resize
            self.after_cancel(self._resize_after) if hasattr(
                self, "_resize_after"
            ) else None
            self._resize_after = self.after(100, self._refresh_image)

    def _refresh_image(self) -> None:
        """Update the displayed image."""
        if not self.pil_image:
            return

        # Step 1: Get container dimensions
        container_width = self.image_container.winfo_width()
        container_height = self.image_container.winfo_height()

        # Use minimum sizes for initialization
        if container_width < 10:
            container_width = self.IMAGE_CONTAINER_WIDTH
        if container_height < 10:
            container_height = self.IMAGE_CONTAINER_HEIGHT

        # Add padding to ensure there's always some margin
        padding = 20  # pixels of padding on each side
        available_width = container_width - (padding * 2)
        available_height = container_height - (padding * 2)

        # Step 2: Calculate target dimensions that maintain aspect ratio
        img_aspect = self.image_width / self.image_height

        # Calculate dimensions that fit within BOTH width and height constraints
        width_constrained = available_width
        height_from_width = int(width_constrained / img_aspect)

        height_constrained = available_height
        width_from_height = int(height_constrained * img_aspect)

        # Use whichever is smaller to ensure image fits completely
        if height_from_width <= available_height:
            display_width = width_constrained
            display_height = height_from_width
        else:
            display_width = width_from_height
            display_height = height_constrained

        # Prevent upscaling beyond original dimensions
        if (
            display_width > self.image_width
            or display_height > self.image_height
        ):
            # Scale down proportionally
            scale = min(
                self.image_width / display_width,
                self.image_height / display_height,
            )
            display_width = int(display_width * scale)
            display_height = int(display_height * scale)

        # Step 3: Store final dimensions for mouse coordinate calculations
        self.display_width = display_width
        self.display_height = display_height

        # Step 4: Center the image in the container
        self.left_offset = (container_width - display_width) // 2
        self.top_offset = (container_height - display_height) // 2

        # Step 5: Create a blank canvas of container size
        canvas = Image.new(
            "RGB", (container_width, container_height), (32, 32, 32)
        )

        # Step 6: Resize image to display dimensions
        if (
            display_width > 0 and display_height > 0
        ):  # Avoid zero-size errors
            resized_image = self.pil_image.resize(
                (display_width, display_height), Image.Resampling.LANCZOS
            )
        else:
            # Fallback if we have invalid dimensions
            resized_image = self.pil_image.copy()
            print("Warning: Invalid display dimensions calculated")

        # Step 7: Handle mask if present
        if self.pil_mask:
            if display_width > 0 and display_height > 0:
                resized_mask = self.pil_mask.resize(
                    (display_width, display_height),
                    Image.Resampling.NEAREST,
                )
            else:
                resized_mask = self.pil_mask.copy()

            if self.display_only_mask:
                final_image = resized_mask
            else:
                # Blend image with mask
                np_image = (
                    np.array(resized_image, dtype=np.float32) / 255.0
                )
                np_mask = np.array(resized_mask, dtype=np.float32) / 255.0

                # Apply minimum opacity
                if np.min(np_mask) == 0:  # Common case
                    np_mask = (
                        np_mask * (1.0 - self.MASK_MIN_OPACITY)
                        + self.MASK_MIN_OPACITY
                    )
                elif np.min(np_mask) < 1:
                    min_val = np.min(np_mask)
                    np_mask = (np_mask - min_val) / (1.0 - min_val) * (
                        1.0 - self.MASK_MIN_OPACITY
                    ) + self.MASK_MIN_OPACITY

                # Apply mask
                np_result = (np_image * np_mask * 255.0).astype(np.uint8)
                final_image = Image.fromarray(np_result, mode="RGB")
        else:
            final_image = resized_image

        # Step 8: Paste image onto canvas
        canvas.paste(final_image, (self.left_offset, self.top_offset))

        # Step 9: Update the displayed image with BOTH parameters set correctly
        self.image.configure(
            light_image=canvas, size=(container_width, container_height)
        )

    def _refresh_caption(self) -> None:
        """Update the displayed caption."""
        self.caption_entry.delete(0, "end")
        self.caption_entry.insert(
            0, self.caption_lines[self.current_caption_line]
        )

    def _clear_ui(self) -> None:
        """Clear the UI when no image is selected."""
        # Create an empty image
        empty_image = Image.new(
            "RGB",
            (self.IMAGE_CONTAINER_WIDTH, self.IMAGE_CONTAINER_HEIGHT),
            (32, 32, 32),
        )
        self.image.configure(light_image=empty_image)

        # Clear caption
        self.caption_entry.delete(0, "end")

        # Reset state
        self.pil_image = None
        self.pil_mask = None
        self.caption_lines = [""] * 5
        self.current_caption_line = 0
        self.caption_line_var.set(self.caption_line_values[0])

    def _handle_mask_edit_start(self, event) -> None:
        """Handle the start of mask editing (mouse button press)."""
        if not self._can_edit_mask(event):
            return

        # Set the editing flag
        self.is_editing = True
        self.edit_started = False

        # Process the first edit point
        self._handle_mask_edit(event)

    def _handle_mask_edit_end(self, event) -> None:
        """Handle the end of mask editing (mouse button release)."""
        if not self.is_editing:
            return

        # Clear the editing flag
        self.is_editing = False

        # Ensure we process the final position
        self._handle_mask_edit(event)

    def _handle_mask_edit(self, event) -> None:
        """Handle mask editing events."""
        if not self._can_edit_mask(event):
            return

        # Get coordinates
        start_x, start_y, end_x, end_y = self._get_edit_coordinates(event)

        # Determine if mouse button is pressed
        is_left = bool(
            event.state & 0x0100 or event.num == 1
        )  # Left mouse
        is_right = bool(
            event.state & 0x0400 or event.num == 3
        )  # Right mouse

        # If this is motion without a mouse button pressed, just track position
        if not (is_left or is_right) and not self.is_editing:
            return

        # Apply mask edit based on current mode
        if self.mask_editing_mode == "draw":
            self._draw_mask(
                start_x, start_y, end_x, end_y, is_left, is_right
            )
        elif self.mask_editing_mode == "fill":
            self._fill_mask(start_x, start_y, is_left, is_right)

    def _can_edit_mask(self, event) -> bool:
        """Check if mask editing is allowed."""
        return (
            self.enable_mask_editing_var.get()
            and event.widget == self.image_label.children["!label"]
            and self.pil_image is not None
            and 0 <= self.current_image_index < len(self.image_rel_paths)
        )

    def _get_edit_coordinates(self, event) -> tuple[int, int, int, int]:
        """Calculate image coordinates for mask editing."""
        # Get scaling factor
        display_scaling = ScalingTracker.get_window_scaling(self)

        # Adjust coordinates for scaling and container offsets
        event_x = event.x / display_scaling
        event_y = event.y / display_scaling

        # Translate to image-local coordinates
        image_x = event_x - self.left_offset
        image_y = event_y - self.top_offset

        # Check if within image boundaries
        if not (
            0 <= image_x < self.display_width
            and 0 <= image_y < self.display_height
        ):
            return 0, 0, 0, 0

        # Convert to original image coordinates
        start_x = int(image_x / self.display_width * self.image_width)
        start_y = int(image_y / self.display_height * self.image_height)

        # Calculate previous position for continuous drawing
        if hasattr(self, "mask_draw_x") and hasattr(self, "mask_draw_y"):
            # Get image-local coordinates from stored raw coordinates
            prev_image_x = self.mask_draw_x - self.left_offset
            prev_image_y = self.mask_draw_y - self.top_offset

            if (
                0 <= prev_image_x < self.display_width
                and 0 <= prev_image_y < self.display_height
            ):
                end_x = int(
                    prev_image_x / self.display_width * self.image_width
                )
                end_y = int(
                    prev_image_y / self.display_height * self.image_height
                )
            else:
                end_x, end_y = start_x, start_y
        else:
            end_x, end_y = start_x, start_y

        # Store current raw position (not image-local)
        self.mask_draw_x = event_x
        self.mask_draw_y = event_y

        return start_x, start_y, end_x, end_y

    def _get_brush_color(
        self, is_left: bool
    ) -> tuple[int, int, int] | None:
        """Get brush color based on operation."""
        if is_left:  # Add to mask
            try:
                opacity = float(self.brush_opacity_entry.get())
                opacity = max(
                    0.0, min(1.0, opacity)
                )  # Clamp between 0 and 1
            except (ValueError, TypeError):
                opacity = 1.0

            rgb_value = int(opacity * 255)
            return (rgb_value, rgb_value, rgb_value)
        elif not is_left:  # Remove from mask
            return (0, 0, 0)

        return None

    def _ensure_mask_exists(self, adding_to_mask: bool) -> None:
        """Create a mask if none exists."""
        if self.pil_mask is None:
            color = (0, 0, 0) if adding_to_mask else (255, 255, 255)
            self.pil_mask = Image.new(
                "RGB", (self.image_width, self.image_height), color=color
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
        """Draw on the mask."""
        color = self._get_brush_color(is_left)
        if color is None:
            return

        adding_to_mask = is_left
        self._ensure_mask_exists(adding_to_mask)

        # Only save to history at the start of an edit
        if not self.edit_started:
            self._save_mask_to_history()
            self.edit_started = True

        # Calculate brush radius
        radius = int(
            self.mask_draw_radius
            * max(self.pil_mask.width, self.pil_mask.height)
        )

        # Create draw object
        draw = ImageDraw.Draw(self.pil_mask)

        # Draw line between points
        draw.line(
            (start_x, start_y, end_x, end_y),
            fill=color,
            width=2 * radius + 1,
        )

        # Draw end caps for smooth lines
        draw.ellipse(
            (
                start_x - radius,
                start_y - radius,
                start_x + radius,
                start_y + radius,
            ),
            fill=color,
        )
        draw.ellipse(
            (
                end_x - radius,
                end_y - radius,
                end_x + radius,
                end_y + radius,
            ),
            fill=color,
        )

        # Update display
        self._refresh_image()

    def _fill_mask(
        self, start_x: int, start_y: int, is_left: bool, is_right: bool
    ) -> None:
        """Fill connected region of the mask."""
        color = self._get_brush_color(is_left)
        if color is None:
            return

        adding_to_mask = is_left
        self._ensure_mask_exists(adding_to_mask)

        # Fill operations are always a single action
        self._save_mask_to_history()
        self.edit_started = True  # Ensure no further history is saved

        # Use OpenCV for efficient flood fill
        np_mask = np.array(self.pil_mask, dtype=np.uint8)
        cv2.floodFill(np_mask, None, (start_x, start_y), color)
        self.pil_mask = Image.fromarray(np_mask, "RGB")

        # Update display
        self._refresh_image()

    def _fill_mask(
        self, start_x: int, start_y: int, is_left: bool, is_right: bool
    ) -> None:
        """Fill connected region of the mask."""
        color = self._get_brush_color(is_left)
        if color is None:
            return

        adding_to_mask = is_left
        self._ensure_mask_exists(adding_to_mask)

        # Save current state to history before modification
        self._save_mask_to_history()

        # Use OpenCV for efficient flood fill
        np_mask = np.array(self.pil_mask, dtype=np.uint8)
        cv2.floodFill(np_mask, None, (start_x, start_y), color)
        self.pil_mask = Image.fromarray(np_mask, "RGB")

        # Update display
        self._refresh_image()

    def _save_mask_to_history(self) -> None:
        """Save current mask state to history before modification."""
        if self.pil_mask is None:
            return

        # Create a copy of the current mask
        current_mask = self.pil_mask.copy()

        # If we're not at the end of history, discard future states
        if self.mask_history_position < len(self.mask_history) - 1:
            self.mask_history = self.mask_history[
                : self.mask_history_position + 1
            ]

        # Add current state to history
        self.mask_history.append(current_mask)

        # Enforce history size limit
        if len(self.mask_history) > self.mask_history_limit:
            self.mask_history.pop(0)

        # Update position pointer
        self.mask_history_position = len(self.mask_history) - 1

    def _undo_mask_edit(self, event=None) -> str:
        """Undo the last mask edit."""
        if not self.mask_history or self.mask_history_position <= 0:
            return "break"  # Nothing to undo

        # Move back in history
        self.mask_history_position -= 1

        # Restore the previous state
        self.pil_mask = self.mask_history[
            self.mask_history_position
        ].copy()

        # Update display
        self._refresh_image()

        return "break"  # Prevent default handling

    def _redo_mask_edit(self, event=None) -> str:
        """Redo the previously undone mask edit."""
        if self.mask_history_position >= len(self.mask_history) - 1:
            return "break"  # Nothing to redo

        # Move forward in history
        self.mask_history_position += 1

        # Restore the next state
        self.pil_mask = self.mask_history[
            self.mask_history_position
        ].copy()

        # Update display
        self._refresh_image()

        return "break"  # Prevent default handling

    def _is_supported_image(self, filename: str) -> bool:
        """Check if the file is a supported image format."""
        name, ext = os.path.splitext(filename)
        return path_util.is_supported_image_extension(
            ext
        ) and not name.endswith("-masklabel")

    def _adjust_brush_size(self, delta: float, raw_event) -> None:
        """Adjust brush size based on mouse wheel movement."""
        # Apply non-linear scaling for smoother adjustments
        multiplier = 1.0 + (delta * 0.05)
        self.mask_draw_radius = max(
            0.0025, self.mask_draw_radius * multiplier
        )

    def _open_directory(self) -> None:
        """Open a directory selection dialog."""
        new_dir = filedialog.askdirectory()
        if new_dir:
            self.dir = new_dir
            self.load_directory()

    def _open_mask_window(self) -> None:
        """Open the automatic mask generation window."""
        if not self.dir:
            return

        dialog = GenerateMasksWindow(
            self, self.dir, self.config_ui_data["include_subdirectories"]
        )
        self.wait_window(dialog)
        # Refresh current image to show newly generated mask if any
        if 0 <= self.current_image_index < len(self.image_rel_paths):
            self._switch_to_image(self.current_image_index)

    def _open_caption_window(self) -> None:
        """Open the automatic caption generation window."""
        if not self.dir:
            return

        dialog = GenerateCaptionsWindow(
            self, self.dir, self.config_ui_data["include_subdirectories"]
        )
        self.wait_window(dialog)
        # Refresh current image to show newly generated caption if any
        if 0 <= self.current_image_index < len(self.image_rel_paths):
            self._switch_to_image(self.current_image_index)

    def _open_in_explorer(self) -> None:
        """Open the current image in file explorer."""
        if not (0 <= self.current_image_index < len(self.image_rel_paths)):
            return

        image_path = os.path.join(
            self.dir, self.image_rel_paths[self.current_image_index]
        )
        image_path = os.path.normpath(os.path.realpath(image_path))

        try:
            if platform.system() == "Windows":
                subprocess.Popen(f"explorer /select,{image_path}")
            elif platform.system() == "Darwin":  # macOS
                subprocess.Popen(["open", "-R", image_path])
            else:  # Linux
                subprocess.Popen(["xdg-open", os.path.dirname(image_path)])
        except Exception as e:
            print(f"Error opening file in explorer: {e}")

    def _show_help(self) -> None:
        """Display help information."""
        print(self.help_text)

    def _save_changes(self, event=None) -> None:
        """Save the current image's mask and caption."""
        if not (0 <= self.current_image_index < len(self.image_rel_paths)):
            return

        image_path = os.path.join(
            self.dir, self.image_rel_paths[self.current_image_index]
        )

        # Save mask if exists
        if self.pil_mask:
            mask_path = os.path.splitext(image_path)[0] + "-masklabel.png"
            try:
                self.pil_mask.save(mask_path)
                print(f"Saved mask to {mask_path}")
            except Exception as e:
                print(f"Error saving mask: {e}")

        # Update current caption line with text from entry
        current_text = self.caption_entry.get()
        self.caption_lines[self.current_caption_line] = current_text

        # Save caption
        caption_path = os.path.splitext(image_path)[0] + ".txt"
        try:
            # Filter out empty lines at the end
            non_empty_lines = []
            for line in self.caption_lines:
                if (
                    line or non_empty_lines
                ):  # Keep empty lines in the middle
                    non_empty_lines.append(line)

            with open(caption_path, "w", encoding="utf-8") as f:
                f.write("\n".join(non_empty_lines))

            print(f"Saved caption to {caption_path}")
        except Exception as e:
            print(f"Error saving caption: {e}")

    def _next_image(self, event=None) -> None:
        """Navigate to the next image."""
        if len(self.image_rel_paths) > 0 and (
            self.current_image_index + 1
        ) < len(self.image_rel_paths):
            self._save_changes()  # Save current changes before moving
            self._switch_to_image(self.current_image_index + 1)
        return "break"  # Prevent default event handling

    def _previous_image(self, event=None) -> None:
        """Navigate to the previous image."""
        if (
            len(self.image_rel_paths) > 0
            and (self.current_image_index - 1) >= 0
        ):
            self._save_changes()  # Save current changes before moving
            self._switch_to_image(self.current_image_index - 1)
        return "break"  # Prevent default event handling

    def _next_caption_line(self, event=None) -> None:
        """Switch to the next caption line."""
        # Save current line text
        current_text = self.caption_entry.get()
        self.caption_lines[self.current_caption_line] = current_text

        # Move to next line (cycling through the available lines)
        self.current_caption_line = (self.current_caption_line + 1) % len(
            self.caption_lines
        )

        # Update UI
        self.caption_line_var.set(
            self.caption_line_values[self.current_caption_line]
        )
        self._refresh_caption()

        return "break"  # Prevent default tab behavior

    def _on_caption_line_changed(self, value: str) -> None:
        """Handle caption line selection change."""
        # Save current line text
        current_text = self.caption_entry.get()
        self.caption_lines[self.current_caption_line] = current_text

        # Get new line index from value string
        try:
            # Extract line number from string like "Line 1"
            line_number = int(value.split(" ")[1]) - 1
            self.current_caption_line = line_number
        except (ValueError, IndexError):
            self.current_caption_line = 0

        # Update displayed text
        self._refresh_caption()

    def _toggle_mask_display(self, event=None) -> None:
        """Toggle between showing image with mask or just the mask."""
        self.display_only_mask = not self.display_only_mask
        self._refresh_image()

        return "break"  # Prevent default event handling

    def _draw_mask_mode(self, event=None) -> None:
        """Switch to draw mask mode."""
        self.mask_editing_mode = "draw"
        return "break" if event else None

    def _fill_mask_mode(self, event=None) -> None:
        """Switch to fill mask mode."""
        self.mask_editing_mode = "fill"
        return "break" if event else None

    def load_masking_model(self, model):
        self.captioning_model = None

        if model == "ClipSeg":
            if self.masking_model is None or not isinstance(
                self.masking_model, ClipSegModel
            ):
                print("loading ClipSeg model, this may take a while")
                self.masking_model = ClipSegModel(
                    default_device, torch.float32
                )
        elif model == "Rembg":
            if self.masking_model is None or not isinstance(
                self.masking_model, RembgModel
            ):
                print("loading Rembg model, this may take a while")
                self.masking_model = RembgModel(
                    default_device, torch.float32
                )
        elif model == "Rembg-Human":
            if self.masking_model is None or not isinstance(
                self.masking_model, RembgHumanModel
            ):
                print("loading Rembg-Human model, this may take a while")
                self.masking_model = RembgHumanModel(
                    default_device, torch.float32
                )
        elif model == "Hex Color":
            if self.masking_model is None or not isinstance(
                self.masking_model, MaskByColor
            ):
                self.masking_model = MaskByColor(
                    default_device, torch.float32
                )

    def load_captioning_model(self, model):
        self.masking_model = None

        if model == "Blip":
            if self.captioning_model is None or not isinstance(
                self.captioning_model, BlipModel
            ):
                print("loading Blip model, this may take a while")
                self.captioning_model = BlipModel(
                    default_device, torch.float16
                )
        elif model == "Blip2":
            if self.captioning_model is None or not isinstance(
                self.captioning_model, Blip2Model
            ):
                print("loading Blip2 model, this may take a while")
                self.captioning_model = Blip2Model(
                    default_device, torch.float16
                )
        elif model == "WD14 VIT v2":
            if self.captioning_model is None or not isinstance(
                self.captioning_model, WDModel
            ):
                print("loading WD14_VIT_v2 model, this may take a while")
                self.captioning_model = WDModel(
                    default_device, torch.float16
                )
