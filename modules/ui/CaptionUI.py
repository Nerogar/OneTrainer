import os
import platform
import subprocess
from collections.abc import Callable
from enum import Enum
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
from customtkinter import ScalingTracker, ThemeManager
from PIL import Image, ImageDraw

RGBColor: TypeAlias = tuple[int, int, int]
ImageCoordinates: TypeAlias = tuple[int, int, int, int]

def scan_for_supported_images(
    directory: str, include_subdirs: bool, is_supported: Callable[[str], bool]
) -> list[str]:
    if include_subdirs:
        results = []
        for root, _, files in os.walk(directory):
            results.extend(
                os.path.relpath(os.path.join(root, filename), directory)
                for filename in files if is_supported(filename)
            )
    else:
        results = [
            filename for filename in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, filename)) and is_supported(filename)
        ]
    results.sort()
    return results


class CaptionUI(ctk.CTkToplevel):
    WINDOW_WIDTH = 1018
    WINDOW_HEIGHT = 768

    FILE_LIST_WIDTH = 250
    MASK_MIN_OPACITY = 0.3
    DEFAULT_BRUSH_SIZE = 0.01

    def __init__(self, parent, initial_dir: str | None = None, include_subdirectories: bool = False, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.dir = initial_dir
        self.config_ui_data = {"include_subdirectories": include_subdirectories}
        self.config_ui_state = UIState(self, self.config_ui_data)

        self.image_rel_paths: list[str] = []
        self.current_image_index = -1
        self.pil_image: Image.Image | None = None
        self.pil_mask: Image.Image | None = None
        self.image_width = 0
        self.image_height = 0

        self.caption_lines: list[str] = []
        self.current_caption_line = 0

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
            self.file_manager.load_directory()

    def _setup_window(self) -> None:
        self.title("OneTrainer Image Editor")
        self.geometry(f"{self.WINDOW_WIDTH}x{self.WINDOW_HEIGHT}")
        self.resizable(True, True)
        self.wait_visibility()
        self.focus_set()
        self.help_text = (
            "Keyboard shortcuts:\n"
            "Up/Down arrows: Navigate between images\n"
            "Tab: Switch between caption lines\n"
            "Return: Save changes\n"
            "Ctrl+M: Toggle mask display\n"
            "Ctrl+D: Switch to draw mode\n"
            "Ctrl+F: Switch to fill mode\n\n"
            "When editing masks:\n"
            "Left click: Add to mask\n"
            "Right click: Remove from mask\n"
            "Mouse wheel: Adjust brush size"
        )

    def _create_layout(self) -> None:
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.enable_mask_editing_var = ctk.BooleanVar(value=False)
        self._create_top_bar()
        self._create_main_content()

    def _create_icon_button(
        self, parent, row: int, column: int, text: str, command, icon_name: str, tooltip: str
    ) -> None:
        icon = load_icon(icon_name, (24, 24))
        components.icon_button(parent, row, column, text, command, image=icon, tooltip=tooltip)

    def _create_top_bar(self) -> None:
        top_frame = ctk.CTkFrame(self, corner_radius=0)
        top_frame.grid(row=0, column=0, sticky="new")
        self._create_icon_button(top_frame, 0, 0, "Load", self.file_manager.open_directory, "load", "Load a directory of images")
        self._create_icon_button(top_frame, 0, 1, "Auto-Mask", self._open_mask_window, "auto-mask", "Generate masks automatically")
        self._create_icon_button(top_frame, 0, 2, "Auto-Caption", self._open_caption_window, "auto-caption", "Generate captions automatically")
        if platform.system() == "Windows":
            self._create_icon_button(top_frame, 0, 3, "Explorer", self.file_manager.open_in_explorer, "explorer", "Open in File Explorer")
        components.switch(top_frame, 0, 4, self.config_ui_state, "include_subdirectories",
                          text="Include Subdirs", tooltip="Include subdirectories when loading images")
        top_frame.grid_columnconfigure(5, weight=1)
        self._create_icon_button(top_frame, 0, 6, "Help", self._show_help, "help", self.help_text)

    def _create_main_content(self) -> None:
        main_frame = ctk.CTkFrame(self, fg_color="#242424")
        main_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=10)
        main_frame.grid_columnconfigure(0, weight=0)
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)
        self._create_file_list(main_frame)
        self._create_editor_panel(main_frame)

    def _create_file_list(self, parent) -> None:
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
        header_frame.grid(row=0, column=0, sticky="ew", padx=2, pady=(2, 4))
        header_frame.grid_columnconfigure(1, weight=1)
        folder_icon = load_icon("folder-tree", (20, 20))
        ctk.CTkLabel(header_frame, text="", image=folder_icon).grid(row=0, column=0, padx=(5, 3), pady=5)
        self.folder_name_label = ctk.CTkLabel(header_frame, text="No folder selected", anchor="w",
                                              font=("Segoe UI", 12, "bold"))
        self.folder_name_label.grid(row=0, column=1, sticky="ew", padx=0, pady=5)
        self.image_labels = []

    def _create_editor_panel(self, parent) -> None:
        editor_frame = ctk.CTkFrame(parent, fg_color="#242424")
        editor_frame.grid(row=0, column=1, sticky="nsew")
        editor_frame.grid_columnconfigure(0, weight=1)
        editor_frame.grid_rowconfigure(0, weight=0)
        editor_frame.grid_rowconfigure(1, weight=1)
        editor_frame.grid_rowconfigure(2, weight=0)
        self._create_tools_bar(editor_frame)
        self._create_image_container(editor_frame)
        self._create_caption_area(editor_frame)

    def _create_tools_bar(self, parent) -> None:
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
        self._create_icon_button(tools_frame, 0, 0, "Draw", self.mask_editor.draw_mask_mode, icons["draw"], "Draw mask with brush")
        self._create_icon_button(tools_frame, 0, 1, "Fill", self.mask_editor.fill_mask_mode, icons["fill"], "Fill areas of the mask")
        ctk.CTkCheckBox(tools_frame, text="Enable Editing",
                        variable=self.enable_mask_editing_var).grid(row=0, column=2, padx=5, pady=2)
        self.brush_opacity_entry = ctk.CTkEntry(tools_frame, width=40)
        self.brush_opacity_entry.insert(0, "1.0")
        self.brush_opacity_entry.grid(row=0, column=3, padx=5, pady=2)
        ctk.CTkLabel(tools_frame, text="Brush Opacity").grid(row=0, column=4, padx=2, pady=2)
        self._create_icon_button(tools_frame, 0, 6, "", self.file_manager.save_changes, icons["save"], "Save changes")
        self._create_icon_button(tools_frame, 0, 7, "", lambda: self.mask_editor.undo_mask_edit(), icons["undo"], "Undo last edit")
        self._create_icon_button(tools_frame, 0, 8, "", lambda: self.mask_editor.redo_mask_edit(), icons["redo"], "Redo last undone edit")

    def _create_image_container(self, parent) -> None:
        self.image_container = ctk.CTkFrame(parent, fg_color="#242424")
        self.image_container.grid(row=1, column=0, sticky="nsew")
        self.image_container.grid_columnconfigure(0, weight=1)
        self.image_container.grid_rowconfigure(0, weight=1)
        default_width = 800
        default_height = 600
        self.image = ctk.CTkImage(
            light_image=Image.new("RGB", (default_width, default_height), (32, 32, 32)),
            size=(default_width, default_height),
        )
        self.image_label = ctk.CTkLabel(self.image_container, text="", image=self.image)
        self.image_label.grid(row=0, column=0, sticky="nsew")
        self.after(100, self.image_handler.update_image_container_size)
        self.image_label.bind("<Motion>", self.mask_editor.handle_mask_edit)
        self.image_label.bind("<Button-1>", self.mask_editor.handle_mask_edit_start)
        self.image_label.bind("<ButtonRelease-1>", self.mask_editor.handle_mask_edit_end)
        self.image_label.bind("<Button-3>", self.mask_editor.handle_mask_edit_start)
        self.image_label.bind("<ButtonRelease-3>", self.mask_editor.handle_mask_edit_end)
        self.bind("<Configure>", self.image_handler.on_resize)
        bind_mousewheel(self.image_label, {self.image_label.children["!label"]}, self.mask_editor.adjust_brush_size)

    def _create_caption_area(self, parent) -> None:
        caption_frame = ctk.CTkFrame(parent, fg_color="#333333")
        caption_frame.grid(row=2, column=0, sticky="sew")
        caption_frame.grid_columnconfigure(0, weight=0)
        caption_frame.grid_columnconfigure(1, weight=1)
        self.caption_line_values = [f"Line {i}" for i in range(1, 6)]
        self.caption_line_var = ctk.StringVar(value=self.caption_line_values[0])
        ctk.CTkOptionMenu(caption_frame, values=self.caption_line_values,
                          variable=self.caption_line_var,
                          command=self.caption_manager.on_caption_line_changed).grid(row=0, column=0, padx=5, pady=5)
        self.caption_entry = ctk.CTkEntry(caption_frame)
        self.caption_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self._bind_key_events(self.caption_entry)

    def _bind_key_events(self, component) -> None:
        component.bind("<Down>", self.navigation_manager.next_image)
        component.bind("<Up>", self.navigation_manager.previous_image)
        component.bind("<Return>", self.file_manager.save_changes)
        component.bind("<Tab>", self.caption_manager.next_caption_line)
        component.bind("<Control-m>", self.mask_editor.toggle_mask_display)
        component.bind("<Control-d>", self.mask_editor.draw_mask_mode)
        component.bind("<Control-f>", self.mask_editor.fill_mask_mode)

    def _update_file_list(self) -> None:
        for widget in self.file_list.winfo_children():
            widget.destroy()
        self.image_labels = []
        header_frame = ctk.CTkFrame(self.file_list)
        header_frame.grid(row=0, column=0, sticky="ew", padx=2, pady=(2, 4))
        header_frame.grid_columnconfigure(1, weight=1)
        folder_icon = load_icon("folder-tree", (20, 20))
        ctk.CTkLabel(header_frame, text="", image=folder_icon).grid(row=0, column=0, padx=(5, 3), pady=5)
        folder_path = os.path.abspath(self.dir) if self.dir else "No folder selected"
        self.folder_name_label = ctk.CTkLabel(header_frame, text=folder_path, anchor="w",
                                              font=("Segoe UI", 12, "bold"),
                                              wraplength=self.FILE_LIST_WIDTH - 35)
        self.folder_name_label.grid(row=0, column=1, sticky="ew", padx=0, pady=5)
        for i, filename in enumerate(self.image_rel_paths):
            safe_filename = filename if isinstance(filename, str) else filename.decode("utf-8", errors="replace")
            label = ctk.CTkLabel(self.file_list, text=safe_filename, wraplength=self.FILE_LIST_WIDTH - 20,
                                  font=("Segoe UI", 11))
            label.bind("<Button-1>", lambda e, idx=i: self.navigation_manager.switch_to_image(idx))
            label.grid(row=i + 1, column=0, sticky="w", padx=2, pady=(1 if i == 0 else 2, 2))
            self.image_labels.append(label)
        self.after(100, self._update_scrollbar_visibility)

    def _update_scrollbar_visibility(self) -> None:
        try:
            if hasattr(self.file_list, "_scrollbar_frame") and hasattr(self.file_list, "_parent_canvas"):
                canvas = self.file_list._parent_canvas
                inner_frame = self.file_list._scrollbar_frame
                if inner_frame.winfo_reqheight() > canvas.winfo_height():
                    self._file_list_scrollbar.grid()
                else:
                    self._file_list_scrollbar.grid_remove()
            else:
                self._file_list_scrollbar.grid() if len(self.image_rel_paths) > 15 else self._file_list_scrollbar.grid_remove()
        except Exception as e:
            self._file_list_scrollbar.grid()
            print(f"Note: Could not adjust scrollbar visibility: {e}")

    def refresh_ui(self) -> None:
        if self.pil_image:
            self.image_handler.refresh_image()
            self.caption_manager.refresh_caption()

    def clear_ui(self) -> None:
        empty_image = Image.new("RGB", (self.IMAGE_CONTAINER_WIDTH, self.IMAGE_CONTAINER_HEIGHT), (32, 32, 32))
        self.image.configure(light_image=empty_image)
        self.caption_entry.delete(0, "end")
        self.pil_image = None
        self.pil_mask = None
        self.caption_lines = [""] * 5
        self.current_caption_line = 0
        self.caption_line_var.set(self.caption_line_values[0])

    def _open_caption_window(self) -> None:
        if self.dir:
            dialog = GenerateCaptionsWindow(self, self.dir, self.config_ui_data["include_subdirectories"])
            self.wait_window(dialog)
            if 0 <= self.current_image_index < len(self.image_rel_paths):
                self.navigation_manager.switch_to_image(self.current_image_index)

    def _open_mask_window(self) -> None:
        if self.dir:
            dialog = GenerateMasksWindow(self, self.dir, self.config_ui_data["include_subdirectories"])
            self.wait_window(dialog)
            if 0 <= self.current_image_index < len(self.image_rel_paths):
                self.navigation_manager.switch_to_image(self.current_image_index)

    def _show_help(self) -> None:
        print(self.help_text)


class ImageHandler:
    def __init__(self, parent) -> None:
        self.parent = parent

    def _calculate_display_dimensions(self) -> tuple[int, int, int, int]:
        # Get actual container dimensions with reasonable defaults
        container_width = max(
            10, self.parent.image_container.winfo_width()
        )
        container_height = max(
            10, self.parent.image_container.winfo_height()
        )

        # Use reasonable defaults if container is too small
        if container_width <= 10:
            container_width = 800
        if container_height <= 10:
            container_height = 600

        # Original image dimensions
        image_width = self.parent.image_width
        image_height = self.parent.image_height

        if image_width <= 0 or image_height <= 0:
            return 0, 0, 0, 0

        # Calculate padding
        padding = 20
        available_width = container_width - (padding * 2)
        available_height = container_height - (padding * 2)

        # Calculate a single scale factor that works for both dimensions
        # This ensures aspect ratio is preserved correctly
        scale = min(
            available_width / image_width, available_height / image_height
        )

        # Never scale up images beyond their original size
        scale = min(scale, 1.0)

        # Apply scale factor to get display dimensions
        display_width = int(image_width * scale)
        display_height = int(image_height * scale)

        # Center the image in the container
        left_offset = (container_width - display_width) // 2
        top_offset = (container_height - display_height) // 2

        return display_width, display_height, left_offset, top_offset

    def refresh_image(self) -> None:
        if not self.parent.pil_image:
            return
        display_width, display_height, left_offset, top_offset = self._calculate_display_dimensions()
        self.parent.display_width = display_width
        self.parent.display_height = display_height
        self.parent.left_offset = left_offset
        self.parent.top_offset = top_offset
        container_width = self.parent.image_container.winfo_width()
        container_height = self.parent.image_container.winfo_height()
        canvas = Image.new("RGB", (container_width, container_height), (32, 32, 32))
        resized_image = (self.parent.pil_image.resize((display_width, display_height), Image.Resampling.LANCZOS)
                         if display_width and display_height else self.parent.pil_image.copy())
        if self.parent.pil_mask:
            resized_mask = (self.parent.pil_mask.resize((display_width, display_height), Image.Resampling.NEAREST)
                            if display_width and display_height else self.parent.pil_mask.copy())
            if self.parent.mask_editor.display_only_mask:
                final_image = resized_mask
            else:
                np_image = np.array(resized_image, dtype=np.float32) / 255.0
                np_mask = np.array(resized_mask, dtype=np.float32) / 255.0
                if np.min(np_mask) == 0:
                    np_mask = np_mask * (1.0 - self.parent.MASK_MIN_OPACITY) + self.parent.MASK_MIN_OPACITY
                elif np.min(np_mask) < 1:
                    min_val = np.min(np_mask)
                    np_mask = (np_mask - min_val) / (1.0 - min_val) * (1.0 - self.parent.MASK_MIN_OPACITY) + self.parent.MASK_MIN_OPACITY
                np_result = (np_image * np_mask * 255.0).astype(np.uint8)
                final_image = Image.fromarray(np_result, mode="RGB")
        else:
            final_image = resized_image
        canvas.paste(final_image, (left_offset, top_offset))
        self.parent.image.configure(light_image=canvas, size=(container_width, container_height))

    def update_image_container_size(self):
        if self.parent.image_container.winfo_width() > 10 and self.parent.image_container.winfo_height() > 10:
            self.refresh_image()
        else:
            self.parent.after(100, self.update_image_container_size)

    def on_resize(self, event=None) -> None:
        if self.parent.pil_image and self.parent.image_container.winfo_ismapped():
            if hasattr(self.parent, "_resize_after"):
                self.parent.after_cancel(self.parent._resize_after)
            self.parent._resize_after = self.parent.after(100, self.refresh_image)

    def load_image_data(self) -> None:
        if not (0 <= self.parent.current_image_index < len(self.parent.image_rel_paths)):
            return
        image_path = os.path.join(self.parent.dir, self.parent.image_rel_paths[self.parent.current_image_index])
        try:
            self.parent.pil_image = Image.open(image_path).convert("RGB")
            self.parent.image_width = self.parent.pil_image.width
            self.parent.image_height = self.parent.pil_image.height
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            self.parent.pil_image = None
        mask_path = os.path.splitext(image_path)[0] + "-masklabel.png"
        if os.path.exists(mask_path):
            try:
                self.parent.pil_mask = Image.open(mask_path).convert("RGB")
                self.parent.mask_editor.reset_for_new_image()
            except Exception as e:
                print(f"Error loading mask {mask_path}: {e}")
                self.parent.pil_mask = None
        else:
            self.parent.pil_mask = None
            self.parent.mask_editor.reset_for_new_image()
        caption_path = os.path.splitext(image_path)[0] + ".txt"
        if os.path.exists(caption_path):
            try:
                with open(caption_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    self.parent.caption_lines = content.split("\n")
                    self.parent.caption_lines.extend([""] * (5 - len(self.parent.caption_lines)))
                    self.parent.caption_lines = self.parent.caption_lines[:5]
            except Exception as e:
                print(f"Error loading caption {caption_path}: {e}")
                self.parent.caption_lines = [""] * 5
        else:
            self.parent.caption_lines = [""] * 5
        self.parent.current_caption_line = 0
        self.parent.caption_line_var.set(self.parent.caption_line_values[0])
        self.parent.caption_entry.delete(0, "end")
        self.parent.caption_entry.insert(0, self.parent.caption_lines[0])

    def is_supported_image(self, filename: str) -> bool:
        name, ext = os.path.splitext(filename)
        return path_util.is_supported_image_extension(ext) and not name.endswith("-masklabel")


class NavigationManager:
    def __init__(self, parent) -> None:
        self.parent = parent

    def switch_to_image(self, index: int) -> None:
        if 0 <= self.parent.current_image_index < len(self.parent.image_labels):
            self.parent.image_labels[self.parent.current_image_index].configure(
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

    def next_image(self, event=None) -> str:
        if self.parent.image_rel_paths and (self.parent.current_image_index + 1) < len(self.parent.image_rel_paths):
            self.parent.file_manager.save_changes()
            self.switch_to_image(self.parent.current_image_index + 1)
        return "break"

    def previous_image(self, event=None) -> str:
        if self.parent.image_rel_paths and (self.parent.current_image_index - 1) >= 0:
            self.parent.file_manager.save_changes()
            self.switch_to_image(self.parent.current_image_index - 1)
        return "break"


class ModelManager:
    def __init__(self, device=None, precision=None):
        self.device = device or default_device
        self.precision = precision or torch.float16
        self.masking_model = None
        self.captioning_model = None

    def load_masking_model(self, model: str):
        self.captioning_model = None
        if model == "ClipSeg" and (self.masking_model is None or not isinstance(self.masking_model, ClipSegModel)):
            print("Loading ClipSeg model, this may take a while")
            self.masking_model = ClipSegModel(self.device, torch.float32)
        elif model == "Rembg" and (self.masking_model is None or not isinstance(self.masking_model, RembgModel)):
            print("Loading Rembg model, this may take a while")
            self.masking_model = RembgModel(self.device, torch.float32)
        elif model == "Rembg-Human" and (self.masking_model is None or not isinstance(self.masking_model, RembgHumanModel)):
            print("Loading Rembg-Human model, this may take a while")
            self.masking_model = RembgHumanModel(self.device, torch.float32)
        elif model == "Hex Color" and (self.masking_model is None or not isinstance(self.masking_model, MaskByColor)):
            self.masking_model = MaskByColor(self.device, torch.float32)

    def load_captioning_model(self, model: str):
        self.masking_model = None
        if model == "Blip" and (self.captioning_model is None or not isinstance(self.captioning_model, BlipModel)):
            print("Loading Blip model, this may take a while")
            self.captioning_model = BlipModel(self.device, self.precision)
        elif model == "Blip2" and (self.captioning_model is None or not isinstance(self.captioning_model, Blip2Model)):
            print("Loading Blip2 model, this may take a while")
            self.captioning_model = Blip2Model(self.device, self.precision)
        elif model == "WD14 VIT v2" and (self.captioning_model is None or not isinstance(self.captioning_model, WDModel)):
            print("Loading WD14_VIT_v2 model, this may take a while")
            self.captioning_model = WDModel(self.device, self.precision)

    def get_masking_model(self):
        return self.masking_model

    def get_captioning_model(self):
        return self.captioning_model


class FileManager:
    def __init__(self, parent) -> None:
        self.parent = parent

    def save_changes(self, event=None) -> None:
        if not (0 <= self.parent.current_image_index < len(self.parent.image_rel_paths)):
            return
        image_path = os.path.join(self.parent.dir, self.parent.image_rel_paths[self.parent.current_image_index])
        if self.parent.pil_mask:
            mask_path = os.path.splitext(image_path)[0] + "-masklabel.png"
            try:
                self.parent.pil_mask.save(mask_path)
                print(f"Saved mask to {mask_path}")
            except Exception as e:
                print(f"Error saving mask: {e}")
        current_text = self.parent.caption_entry.get()
        self.parent.caption_lines[self.parent.current_caption_line] = current_text
        caption_path = os.path.splitext(image_path)[0] + ".txt"
        try:
            non_empty_lines = [line for line in self.parent.caption_lines if line]
            with open(caption_path, "w", encoding="utf-8") as f:
                f.write("\n".join(non_empty_lines))
            print(f"Saved caption to {caption_path}")
        except Exception as e:
            print(f"Error saving caption: {e}")

    def open_in_explorer(self) -> None:
        if not (0 <= self.parent.current_image_index < len(self.parent.image_rel_paths)):
            return
        image_path = os.path.join(self.parent.dir, self.parent.image_rel_paths[self.parent.current_image_index])
        image_path = os.path.normpath(os.path.realpath(image_path))
        try:
            if platform.system() == "Windows":
                subprocess.Popen(f"explorer /select,{image_path}")
            elif platform.system() == "Darwin":
                subprocess.Popen(["open", "-R", image_path])
            else:
                subprocess.Popen(["xdg-open", os.path.dirname(image_path)])
        except Exception as e:
            print(f"Error opening file in explorer: {e}")

    def open_directory(self) -> None:
        new_dir = filedialog.askdirectory()
        if new_dir:
            self.parent.dir = new_dir
            self.load_directory()

    def load_directory(self) -> None:
        if not self.parent.dir:
            return
        folder_name = os.path.basename(os.path.normpath(self.parent.dir))
        self.parent.folder_name_label.configure(text=folder_name)
        include_subdirs = self.parent.config_ui_data["include_subdirectories"]
        self.parent.image_rel_paths = scan_for_supported_images(
            self.parent.dir, include_subdirs, self.parent.image_handler.is_supported_image
        )
        self.parent._update_file_list()
        if self.parent.image_rel_paths:
            self.parent.navigation_manager.switch_to_image(0)
        else:
            self.parent.clear_ui()


class CaptionManager:
    def __init__(self, parent) -> None:
        self.parent = parent

    def refresh_caption(self) -> None:
        self.parent.caption_entry.delete(0, "end")
        self.parent.caption_entry.insert(0, self.parent.caption_lines[self.parent.current_caption_line])

    def next_caption_line(self, event=None) -> None:
        current_text = self.parent.caption_entry.get()
        self.parent.caption_lines[self.parent.current_caption_line] = current_text
        self.parent.current_caption_line = (self.parent.current_caption_line + 1) % len(self.parent.caption_lines)
        self.parent.caption_line_var.set(self.parent.caption_line_values[self.parent.current_caption_line])
        self.refresh_caption()
        return "break"

    def on_caption_line_changed(self, value: str) -> None:
        current_text = self.parent.caption_entry.get()
        self.parent.caption_lines[self.parent.current_caption_line] = current_text
        try:
            line_number = int(value.split(" ")[1]) - 1
            self.parent.current_caption_line = line_number
        except (ValueError, IndexError):
            self.parent.current_caption_line = 0
        self.refresh_caption()


class EditMode(Enum):
    DRAW = "draw"
    FILL = "fill"


class MaskEditor:
    DEFAULT_BRUSH_SIZE = 0.02

    def __init__(self, parent) -> None:
        self.parent = parent
        self.mask_draw_x: float = 0
        self.mask_draw_y: float = 0
        self.mask_draw_radius: float = self.DEFAULT_BRUSH_SIZE
        self.mask_editing_mode: EditMode = EditMode.DRAW
        self.display_only_mask: bool = False
        self.mask_history: list = []
        self.mask_history_position: int = -1
        self.mask_history_limit: int = 6
        self.is_editing: bool = False
        self.edit_started: bool = False
        self._cached_image_dimensions: tuple[int, int] = (0, 0)

    def reset_for_new_image(self) -> None:
        self.mask_history = []
        self.mask_history_position = -1
        self.is_editing = False
        self.edit_started = False
        self._cached_image_dimensions = (self.parent.image_width, self.parent.image_height)
        if self.parent.pil_mask:
            self.mask_history.append(self.parent.pil_mask.copy())
            self.mask_history_position = 0

    def handle_mask_edit_start(self, event) -> None:
        if not self._can_edit_mask(event):
            return
        self.is_editing = True
        self.edit_started = False
        self.handle_mask_edit(event)

    def handle_mask_edit_end(self, event) -> None:
        if not self.is_editing:
            return
        self.is_editing = False
        self.handle_mask_edit(event)
        if self.edit_started:
            self._save_mask_to_history()
            self.edit_started = False

    def handle_mask_edit(self, event) -> None:
        if not self._can_edit_mask(event):
            return
        start_x, start_y, end_x, end_y = self._get_edit_coordinates(event)
        if start_x == end_x == 0 and start_y == end_y == 0:
            return
        is_left = bool(event.state & 0x0100 or event.num == 1)
        is_right = bool(event.state & 0x0400 or event.num == 3)
        if not (is_left or is_right) and not self.is_editing:
            return
        match self.mask_editing_mode:
            case EditMode.DRAW:
                self._draw_mask(start_x, start_y, end_x, end_y, is_left, is_right)
            case EditMode.FILL:
                self._fill_mask(start_x, start_y, is_left, is_right)

    def _can_edit_mask(self, event) -> bool:
        return (
            self.parent.enable_mask_editing_var.get()
            and event.widget == self.parent.image_label.children["!label"]
            and self.parent.pil_image is not None
            and 0 <= self.parent.current_image_index < len(self.parent.image_rel_paths)
        )

    def _get_edit_coordinates(self, event) -> ImageCoordinates:
        display_scaling = ScalingTracker.get_window_scaling(self.parent)
        left_offset = self.parent.left_offset
        top_offset = self.parent.top_offset
        display_width = self.parent.display_width
        display_height = self.parent.display_height
        image_width = self.parent.image_width
        image_height = self.parent.image_height
        event_x = event.x / display_scaling
        event_y = event.y / display_scaling
        image_x = event_x - left_offset
        image_y = event_y - top_offset
        if not (0 <= image_x < display_width and 0 <= image_y < display_height):
            return 0, 0, 0, 0
        start_x = int(image_x * image_width / display_width)
        start_y = int(image_y * image_height / display_height)
        if hasattr(self, "mask_draw_x") and hasattr(self, "mask_draw_y"):
            prev_image_x = self.mask_draw_x - left_offset
            prev_image_y = self.mask_draw_y - top_offset
            if 0 <= prev_image_x < display_width and 0 <= prev_image_y < display_height:
                end_x = int(prev_image_x * image_width / display_width)
                end_y = int(prev_image_y * image_height / display_height)
            else:
                end_x, end_y = start_x, start_y
        else:
            end_x, end_y = start_x, start_y
        self.mask_draw_x = event_x
        self.mask_draw_y = event_y
        return start_x, start_y, end_x, end_y

    def _get_brush_color(self, is_left: bool) -> RGBColor | None:
        if is_left:
            try:
                opacity = float(self.parent.brush_opacity_entry.get())
                opacity = max(0.0, min(1.0, opacity))
            except (ValueError, TypeError):
                opacity = 1.0
            rgb_value = int(opacity * 255)
            return (rgb_value, rgb_value, rgb_value)
        return (0, 0, 0) if not is_left else None

    def _ensure_mask_exists(self, adding_to_mask: bool) -> None:
        if self.parent.pil_mask is None:
            color = (0, 0, 0) if adding_to_mask else (255, 255, 255)
            self.parent.pil_mask = Image.new("RGB", (self.parent.image_width, self.parent.image_height), color=color)

    def _draw_mask(self, start_x: int, start_y: int, end_x: int, end_y: int, is_left: bool, is_right: bool) -> None:
        if color := self._get_brush_color(is_left):
            self._ensure_mask_exists(is_left)
            if not self.edit_started:
                self._save_mask_to_history()
                self.edit_started = True
            max_dimension = max(self.parent.pil_mask.width, self.parent.pil_mask.height)
            radius = int(self.mask_draw_radius * max_dimension)
            if radius <= 0 or (start_x == end_x == start_y == end_y == 0):
                return
            draw = ImageDraw.Draw(self.parent.pil_mask)
            line_width = 2 * radius + 1
            draw.line((start_x, start_y, end_x, end_y), fill=color, width=line_width)
            draw.ellipse((start_x - radius, start_y - radius, start_x + radius, start_y + radius), fill=color)
            if (start_x, start_y) != (end_x, end_y):
                draw.ellipse((end_x - radius, end_y - radius, end_x + radius, end_y + radius), fill=color)
            self.parent.image_handler.refresh_image()

    def _fill_mask(self, start_x: int, start_y: int, is_left: bool, is_right: bool) -> None:
        if color := self._get_brush_color(is_left):
            self._ensure_mask_exists(is_left)
            if not (0 <= start_x < self.parent.image_width and 0 <= start_y < self.parent.image_height):
                return
            self._save_mask_to_history()
            self.edit_started = True
            np_mask = np.array(self.parent.pil_mask, dtype=np.uint8)
            cv2.floodFill(np_mask, None, (start_x, start_y), color)
            self.parent.pil_mask = Image.fromarray(np_mask, "RGB")
            self.parent.image_handler.refresh_image()

    def adjust_brush_size(self, delta: float, raw_event) -> None:
        multiplier = 1.0 + (delta * (0.03 if self.mask_draw_radius < 0.05 else 0.05))
        self.mask_draw_radius = max(0.0025, min(0.5, self.mask_draw_radius * multiplier))

    def _save_mask_to_history(self) -> None:
        if self.parent.pil_mask is None:
            return
        current_mask = self.parent.pil_mask.copy()
        if self.mask_history_position < len(self.mask_history) - 1:
            self.mask_history = self.mask_history[:self.mask_history_position + 1]
        self.mask_history.append(current_mask)
        if len(self.mask_history) > self.mask_history_limit:
            self.mask_history.pop(0)
        self.mask_history_position = len(self.mask_history) - 1

    def undo_mask_edit(self, event=None) -> str | None:
        if not self.mask_history or self.mask_history_position <= 0:
            return "break" if event else None
        self.mask_history_position -= 1
        self.parent.pil_mask = self.mask_history[self.mask_history_position].copy()
        self.parent.image_handler.refresh_image()
        return "break" if event else None

    def redo_mask_edit(self, event=None) -> str | None:
        if self.mask_history_position >= len(self.mask_history) - 1:
            return "break" if event else None
        self.mask_history_position += 1
        self.parent.pil_mask = self.mask_history[self.mask_history_position].copy()
        self.parent.image_handler.refresh_image()
        return "break" if event else None

    def draw_mask_mode(self, event=None) -> str | None:
        self.mask_editing_mode = EditMode.DRAW
        return "break" if event else None

    def fill_mask_mode(self, event=None) -> str | None:
        self.mask_editing_mode = EditMode.FILL
        return "break" if event else None

    def toggle_mask_display(self, event=None) -> str:
        self.display_only_mask = not self.display_only_mask
        self.parent.image_handler.refresh_image()
        return "break"
