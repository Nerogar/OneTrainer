import os
import platform
import subprocess
import traceback
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

_icon_cache = {}


class CaptionUI(ctk.CTkToplevel):
    def __init__(
        self,
        parent,
        initial_dir: str | None,
        initial_include_subdirectories: bool,
        *args,
        **kwargs,
    ):
        ctk.CTkToplevel.__init__(self, parent, *args, **kwargs)

        self.dir = initial_dir
        self.config_ui_data = {
            "include_subdirectories": initial_include_subdirectories
        }
        self.config_ui_state = UIState(self, self.config_ui_data)
        # Fixed image size
        self.image_size = 650

        self.title("OneTrainer")
        self.geometry("900x650")  # Fixed initial size
        self.resizable(True, True)
        self.wait_visibility()
        self.focus_set()

        self.help_text = """
Keyboard shortcuts when focusing on the prompt input field:
Up arrow: previous image
Down arrow: next image
Return: save
Ctrl+M: only show the mask
Ctrl+D: draw mask editing mode
Ctrl+F: fill mask editing mode

When editing masks:
Left click: add mask
Right click: remove mask
Mouse wheel: increase or decrease brush size"""

        self.masking_model = None
        self.captioning_model = None

        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # relative path from self.dir to each image
        self.image_rel_paths = []
        self.current_image_index = -1

        self.top_bar(self)

        self.bottom_frame = ctk.CTkFrame(self)
        self.bottom_frame.grid(row=1, column=0, sticky="nsew")

        self.bottom_frame.grid_rowconfigure(0, weight=1)
        self.bottom_frame.grid_columnconfigure(0, weight=0)
        self.bottom_frame.grid_columnconfigure(1, weight=1)

        self.file_list = None
        self.image_labels = []
        self.file_list_column(self.bottom_frame)

        self.pil_image = None
        self.image_width = 0
        self.image_height = 0
        self.pil_mask = None
        self.mask_draw_x = 0
        self.mask_draw_y = 0
        self.mask_draw_radius = 0.01
        self.display_only_mask = False
        self.image = None
        self.image_label = None
        self.mask_editing_mode = "draw"
        self.enable_mask_editing_var = ctk.BooleanVar()
        self.mask_editing_alpha = None
        self.prompt_var = None
        self.prompt_component = None
        self.content_column(self.bottom_frame)

        self.load_directory()

    def top_bar(self, master):
        top_frame = ctk.CTkFrame(master)
        top_frame.grid(row=0, column=0, sticky="nsew")

        # Load icons for buttons
        icon_size = (24, 24)
        load_icon_img = load_icon("load", icon_size)
        mask_icon_img = load_icon("auto-mask", icon_size)
        caption_icon_img = load_icon("auto-caption", icon_size)
        explorer_icon_img = load_icon("explorer", icon_size)
        help_icon_img = load_icon("help", icon_size)

        # Create buttons with icons and auto-sizing
        components.icon_button(
            top_frame,
            0,
            0,
            "Load",
            self.open_directory,
            image=load_icon_img,
            tooltip="Load a directory of images to edit",
        )
        components.icon_button(
            top_frame,
            0,
            1,
            "Auto-Mask",
            self.open_mask_window,
            image=mask_icon_img,
            tooltip="Open a dialog to automatically generate masks",
        )
        components.icon_button(
            top_frame,
            0,
            2,
            "Auto-Caption",
            self.open_caption_window,
            image=caption_icon_img,
            tooltip="Open a dialog to automatically generate captions",
        )

        if platform.system() == "Windows":
            components.icon_button(
                top_frame,
                0,
                3,
                "Open in Explorer",
                self.open_in_explorer,
                image=explorer_icon_img,
                tooltip="Open file location in Windows Explorer",
            )

        components.switch(
            top_frame,
            0,
            4,
            self.config_ui_state,
            "include_subdirectories",
            text="include subdirs",
            tooltip="Include subdirectories when loading images",
        )

        top_frame.grid_columnconfigure(5, weight=1)

        components.icon_button(
            top_frame,
            0,
            6,
            "Help",
            self.print_help,
            image=help_icon_img,
            tooltip=self.help_text,
        )

    def file_list_column(self, master):
        if self.file_list is not None:
            self.image_labels = []
            self.file_list.destroy()

        self.file_list = ctk.CTkScrollableFrame(master, width=250)
        self.file_list.grid(row=0, column=0, sticky="nsew")

        for i, filename in enumerate(self.image_rel_paths):

            def __create_switch_image(index):
                def __switch_image(event):
                    self.switch_image(index)

                return __switch_image

            # wrapping to cap excessively long filenames
            label = ctk.CTkLabel(
                self.file_list, text=filename, wraplength=230
            )
            label.bind("<Button-1>", __create_switch_image(i))

            self.image_labels.append(label)
            label.grid(row=i, column=0, padx=2, pady=2, sticky="nsw")

    def content_column(self, master):
        image = Image.new("RGBA", (512, 512), (0, 0, 0, 0))

        right_frame = ctk.CTkFrame(master, fg_color="transparent")
        right_frame.grid(row=0, column=1, sticky="nsew")

        # Update grid configuration to let buttons use only the space they need
        right_frame.grid_columnconfigure(0, weight=0)  # Draw button column
        right_frame.grid_columnconfigure(1, weight=0)  # Fill button column
        right_frame.grid_columnconfigure(2, weight=1)  # Enable mask checkbox (expand)
        right_frame.grid_columnconfigure(3, weight=0)  # Alpha textbox (fixed size)
        right_frame.grid_columnconfigure(4, weight=1)  # Alpha label (expand remaining)
        right_frame.grid_rowconfigure(1, weight=1)  # Make image row expandable

        # Reduce padding on buttons and add icons
        draw_icon_img = load_icon("draw", (24, 24))
        fill_icon_img = load_icon("fill", (24, 24))

        components.icon_button(
            right_frame,
            0,
            0,
            "Draw",
            self.draw_mask_editing_mode,
            image=draw_icon_img,
            tooltip="draw a mask using a brush",
        )
        components.icon_button(
            right_frame,
            0,
            1,
            "Fill",
            self.fill_mask_editing_mode,
            image=fill_icon_img,
            tooltip="draw a mask using a fill tool",
        )

        # checkbox to enable mask editing
        self.enable_mask_editing_var = ctk.BooleanVar()
        self.enable_mask_editing_var.set(False)
        enable_mask_editing_checkbox = ctk.CTkCheckBox(
            right_frame,
            text="Enable Editing",
            variable=self.enable_mask_editing_var,
            width=50,
        )
        enable_mask_editing_checkbox.grid(
            row=0, column=2, padx=5, pady=2, sticky="w"
        )

        # mask alpha textbox
        self.mask_editing_alpha = ctk.CTkEntry(
            master=right_frame, width=40, placeholder_text="1.0"
        )
        self.mask_editing_alpha.insert(0, "1.0")
        self.mask_editing_alpha.grid(
            row=0, column=3, sticky="e", padx=2, pady=2
        )
        self.bind_key_events(self.mask_editing_alpha)

        mask_editing_alpha_label = ctk.CTkLabel(
            right_frame, text="Brush Opacity", width=85
        )
        mask_editing_alpha_label.grid(
            row=0, column=4, padx=0, pady=2, sticky="w"
        )

        # Create frame specifically for image containment
        self.image_container = ctk.CTkFrame(
            right_frame, fg_color="transparent"
        )
        self.image_container.grid(
            row=1, column=0, columnspan=5, sticky="nsew", padx=3, pady=3
        )
        self.image_container.grid_rowconfigure(0, weight=1)
        self.image_container.grid_columnconfigure(0, weight=1)

        # image
        self.image = ctk.CTkImage(light_image=image, size=(512, 512))
        self.image_label = ctk.CTkLabel(
            master=self.image_container, text="", image=self.image
        )
        self.image_label.grid(row=0, column=0, sticky="nsew")

        self.image_label.bind("<Motion>", self.edit_mask)
        self.image_label.bind("<Button-1>", self.edit_mask)
        self.image_label.bind("<Button-3>", self.edit_mask)
        bind_mousewheel(
            self.image_label,
            {self.image_label.children["!label"]},
            self.draw_mask_radius,
        )

        # prompt
        self.prompt_var = ctk.StringVar()
        self.prompt_component = ctk.CTkEntry(
            right_frame, textvariable=self.prompt_var
        )
        self.prompt_component.grid(
            row=2, column=0, columnspan=5, pady=5, sticky="new"
        )
        self.bind_key_events(self.prompt_component)
        self.prompt_component.focus_set()

    def bind_key_events(self, component):
        component.bind("<Down>", self.next_image)
        component.bind("<Up>", self.previous_image)
        component.bind("<Return>", self.save)
        component.bind("<Control-m>", self.toggle_mask)
        component.bind("<Control-d>", self.draw_mask_editing_mode)
        component.bind("<Control-f>", self.fill_mask_editing_mode)

    def load_directory(self, include_subdirectories: bool = False):
        self.scan_directory(include_subdirectories)
        self.file_list_column(self.bottom_frame)

        if len(self.image_rel_paths) > 0:
            self.switch_image(0)
        else:
            self.switch_image(-1)

        self.prompt_component.focus_set()

    def scan_directory(self, include_subdirectories: bool = False):
        def __is_supported_image_extension(filename):
            name, ext = os.path.splitext(filename)
            return path_util.is_supported_image_extension(
                ext
            ) and not name.endswith("-masklabel")

        self.image_rel_paths = []

        if not self.dir or not os.path.isdir(self.dir):
            return

        if include_subdirectories:
            for root, _, files in os.walk(self.dir):
                for filename in files:
                    if __is_supported_image_extension(filename):
                        self.image_rel_paths.append(
                            os.path.relpath(
                                os.path.join(root, filename), self.dir
                            )
                        )
        else:
            for _, filename in enumerate(os.listdir(self.dir)):
                if __is_supported_image_extension(filename):
                    self.image_rel_paths.append(
                        os.path.relpath(
                            os.path.join(self.dir, filename), self.dir
                        )
                    )

    def load_image(self):
        image_name = "resources/icons/icon.png"

        if len(
            self.image_rel_paths
        ) > 0 and self.current_image_index < len(self.image_rel_paths):
            image_name = self.image_rel_paths[self.current_image_index]
            image_name = os.path.join(self.dir, image_name)

        try:
            return Image.open(image_name).convert("RGB")
        except Exception:
            print(f"Could not open image {image_name}")

    def load_mask(self):
        if len(
            self.image_rel_paths
        ) > 0 and self.current_image_index < len(self.image_rel_paths):
            image_name = self.image_rel_paths[self.current_image_index]
            mask_name = os.path.splitext(image_name)[0] + "-masklabel.png"
            mask_name = os.path.join(self.dir, mask_name)

            try:
                return Image.open(mask_name).convert("RGB")
            except Exception:
                return None
        else:
            return None

    def load_prompt(self):
        if len(
            self.image_rel_paths
        ) > 0 and self.current_image_index < len(self.image_rel_paths):
            image_name = self.image_rel_paths[self.current_image_index]
            prompt_name = os.path.splitext(image_name)[0] + ".txt"
            prompt_name = os.path.join(self.dir, prompt_name)

            try:
                with open(prompt_name, "r", encoding="utf-8") as f:
                    return f.readlines()[0].strip()
            except Exception:
                return ""
        else:
            return ""

    def previous_image(self, event):
        if (
            len(self.image_rel_paths) > 0
            and (self.current_image_index - 1) >= 0
        ):
            self.switch_image(self.current_image_index - 1)

    def next_image(self, event):
        if len(self.image_rel_paths) > 0 and (
            self.current_image_index + 1
        ) < len(self.image_rel_paths):
            self.switch_image(self.current_image_index + 1)

    def refresh_image(self):
        if not self.pil_image:
            return

        # Container size (fixed for this implementation)
        container_width = 512
        container_height = 512

        # Calculate aspect ratio of original image
        img_aspect = self.image_width / self.image_height

        # Determine the size to fit in container while maintaining aspect ratio
        if img_aspect > 1:  # Wider than tall
            display_width = container_width
            display_height = int(container_width / img_aspect)
        else:  # Taller than wide or square
            display_height = container_height
            display_width = int(container_height * img_aspect)

        # Store display dimensions for mouse coordinate calculations
        self.display_width = display_width
        self.display_height = display_height

        # Calculate centering offsets
        self.left_offset = (container_width - display_width) // 2
        self.top_offset = (container_height - display_height) // 2

        # Create a blank canvas
        blank = Image.new(
            "RGB", (container_width, container_height), color=(32, 32, 32)
        )

        # Resize image for display
        resized_image = self.pil_image.resize(
            (display_width, display_height), Image.Resampling.LANCZOS
        )

        # Create the display image - combining original image with mask if present
        if self.pil_mask:
            resized_pil_mask = self.pil_mask.resize(
                (display_width, display_height), Image.Resampling.NEAREST
            )

            if self.display_only_mask:
                display_image = resized_pil_mask
            else:
                np_image = (
                    np.array(resized_image).astype(np.float32) / 255.0
                )
                np_mask = (
                    np.array(resized_pil_mask).astype(np.float32) / 255.0
                )

                # normalize mask between 0.3 - 1.0 so we can see image underneath
                norm_min = 0.3
                np_mask_min = np_mask.min()
                if np_mask_min == 0:
                    # optimize for common case
                    np_mask = np_mask * (1.0 - norm_min) + norm_min
                elif np_mask_min < 1:
                    # note: min of 1 means we get divide by 0
                    np_mask = (np_mask - np_mask_min) / (
                        1.0 - np_mask_min
                    ) * (1.0 - norm_min) + norm_min

                np_masked_image = (np_image * np_mask * 255.0).astype(
                    np.uint8
                )
                display_image = Image.fromarray(
                    np_masked_image, mode="RGB"
                )
        else:
            display_image = resized_image

        # Paste the resized image onto the centered canvas
        blank.paste(display_image, (self.left_offset, self.top_offset))

        # Update the displayed image with the canvas
        self.image.configure(
            light_image=blank, size=(container_width, container_height)
        )

    def switch_image(self, index):
        if len(self.image_labels) > 0 and self.current_image_index < len(
            self.image_labels
        ):
            self.image_labels[self.current_image_index].configure(
                text_color=ThemeManager.theme["CTkLabel"]["text_color"]
            )

        self.current_image_index = index
        if index >= 0:
            self.image_labels[index].configure(text_color="#FF0000")

            self.pil_image = self.load_image()
            self.pil_mask = self.load_mask()
            prompt = self.load_prompt()

            self.image_width = self.pil_image.width
            self.image_height = self.pil_image.height

            self.refresh_image()
            self.prompt_var.set(prompt)
        else:
            image = Image.new("RGB", (512, 512), (0, 0, 0))
            self.image.configure(light_image=image)

    def draw_mask_radius(self, delta, raw_event):
        # Wheel up = Increase radius. Wheel down = Decrease radius.
        multiplier = 1.0 + (delta * 0.05)
        self.mask_draw_radius = max(
            0.0025, self.mask_draw_radius * multiplier
        )

    def edit_mask(self, event):
        if not self.enable_mask_editing_var.get():
            return

        if event.widget != self.image_label.children["!label"]:
            return

        if len(
            self.image_rel_paths
        ) == 0 or self.current_image_index >= len(self.image_rel_paths):
            return

        # Get the display scaling factor for high DPI displays
        display_scaling = ScalingTracker.get_window_scaling(self)

        # Adjust event coordinates for scaling
        event_x = event.x / display_scaling
        event_y = event.y / display_scaling

        # Account for the centering offset
        event_x -= getattr(self, "left_offset", 0)
        event_y -= getattr(self, "top_offset", 0)

        # Check if the click is within the image boundaries
        if (
            event_x < 0
            or event_x >= getattr(self, "display_width", 0)
            or event_y < 0
            or event_y >= getattr(self, "display_height", 0)
        ):
            return  # Click is outside the actual image area

        # Convert screen coordinates to original image coordinates
        start_x = int((event_x / self.display_width) * self.image_width)
        start_y = int((event_y / self.display_height) * self.image_height)

        # Handle previous position for drawing lines
        if hasattr(self, "mask_draw_x") and hasattr(self, "mask_draw_y"):
            prev_x = self.mask_draw_x - getattr(self, "left_offset", 0)
            prev_y = self.mask_draw_y - getattr(self, "top_offset", 0)

            # Check if previous position was within image
            if (
                0 <= prev_x < self.display_width
                and 0 <= prev_y < self.display_height
            ):
                end_x = int(
                    (prev_x / self.display_width) * self.image_width
                )
                end_y = int(
                    (prev_y / self.display_height) * self.image_height
                )
            else:
                end_x = start_x
                end_y = start_y
        else:
            end_x = start_x
            end_y = start_y

        # Store current position for next event
        self.mask_draw_x = event_x + getattr(self, "left_offset", 0)
        self.mask_draw_y = event_y + getattr(self, "top_offset", 0)

        is_right = False
        is_left = False
        if event.state & 0x0100 or event.num == 1:  # left mouse button
            is_left = True
        elif event.state & 0x0400 or event.num == 3:  # right mouse button
            is_right = True

        if self.mask_editing_mode == "draw":
            self.draw_mask(
                start_x, start_y, end_x, end_y, is_left, is_right
            )
        if self.mask_editing_mode == "fill":
            self.fill_mask(
                start_x, start_y, end_x, end_y, is_left, is_right
            )

    def draw_mask(self, start_x, start_y, end_x, end_y, is_left, is_right):
        color = None

        adding_to_mask = True
        if is_left:
            try:
                alpha = float(self.mask_editing_alpha.get())
            except Exception:
                alpha = 1.0
            rgb_value = int(
                max(0.0, min(alpha, 1.0)) * 255
            )  # max/min stuff to clamp to 0 - 255 range
            color = (rgb_value, rgb_value, rgb_value)

        elif is_right:
            color = (0, 0, 0)
            adding_to_mask = False

        if color is not None:
            if self.pil_mask is None:
                if adding_to_mask:
                    self.pil_mask = Image.new(
                        "RGB",
                        size=(self.image_width, self.image_height),
                        color=(0, 0, 0),
                    )
                else:
                    self.pil_mask = Image.new(
                        "RGB",
                        size=(self.image_width, self.image_height),
                        color=(255, 255, 255),
                    )

            radius = int(
                self.mask_draw_radius
                * max(self.pil_mask.width, self.pil_mask.height)
            )

            draw = ImageDraw.Draw(self.pil_mask)
            draw.line(
                (start_x, start_y, end_x, end_y),
                fill=color,
                width=radius + radius + 1,
            )
            draw.ellipse(
                (
                    start_x - radius,
                    start_y - radius,
                    start_x + radius,
                    start_y + radius,
                ),
                fill=color,
                outline=None,
            )
            draw.ellipse(
                (
                    end_x - radius,
                    end_y - radius,
                    end_x + radius,
                    end_y + radius,
                ),
                fill=color,
                outline=None,
            )

            self.refresh_image()

    def fill_mask(self, start_x, start_y, end_x, end_y, is_left, is_right):
        color = None

        adding_to_mask = True
        if is_left:
            try:
                alpha = float(self.mask_editing_alpha.get())
            except Exception:
                alpha = 1.0
            rgb_value = int(
                max(0.0, min(alpha, 1.0)) * 255
            )  # max/min stuff to clamp to 0 - 255 range
            color = (rgb_value, rgb_value, rgb_value)

        elif is_right:
            color = (0, 0, 0)
            adding_to_mask = False

        if color is not None:
            if self.pil_mask is None:
                if adding_to_mask:
                    self.pil_mask = Image.new(
                        "RGB",
                        size=(self.image_width, self.image_height),
                        color=(0, 0, 0),
                    )
                else:
                    self.pil_mask = Image.new(
                        "RGB",
                        size=(self.image_width, self.image_height),
                        color=(255, 255, 255),
                    )

            np_mask = np.array(self.pil_mask).astype(np.uint8)
            cv2.floodFill(np_mask, None, (start_x, start_y), color)
            self.pil_mask = Image.fromarray(np_mask, "RGB")

            self.refresh_image()

    def save(self, event):
        if len(
            self.image_rel_paths
        ) > 0 and self.current_image_index < len(self.image_rel_paths):
            image_name = self.image_rel_paths[self.current_image_index]

            prompt_name = os.path.splitext(image_name)[0] + ".txt"
            prompt_name = os.path.join(self.dir, prompt_name)

            mask_name = os.path.splitext(image_name)[0] + "-masklabel.png"
            mask_name = os.path.join(self.dir, mask_name)

            try:
                with open(prompt_name, "w", encoding="utf-8") as f:
                    f.write(self.prompt_var.get())
            except Exception:
                return

            if self.pil_mask:
                self.pil_mask.save(mask_name)

    def draw_mask_editing_mode(self, *args):
        self.mask_editing_mode = "draw"

        if args:
            # disable default event
            return "break"
        return None

    def fill_mask_editing_mode(self, *args):
        self.mask_editing_mode = "fill"

    def toggle_mask(self, *args):
        self.display_only_mask = not self.display_only_mask
        self.refresh_image()

    def open_directory(self):
        new_dir = filedialog.askdirectory()

        if new_dir:
            self.dir = new_dir
            self.load_directory(
                include_subdirectories=self.config_ui_data[
                    "include_subdirectories"
                ]
            )

    def open_mask_window(self):
        dialog = GenerateMasksWindow(
            self, self.dir, self.config_ui_data["include_subdirectories"]
        )
        self.wait_window(dialog)
        self.switch_image(self.current_image_index)

    def open_caption_window(self):
        dialog = GenerateCaptionsWindow(
            self, self.dir, self.config_ui_data["include_subdirectories"]
        )
        self.wait_window(dialog)
        self.switch_image(self.current_image_index)

    def open_in_explorer(self):
        try:
            image_name = self.image_rel_paths[self.current_image_index]
            image_name = os.path.realpath(
                os.path.join(self.dir, image_name)
            )
            subprocess.Popen(f"explorer /select,{image_name}")
        except Exception:
            traceback.print_exc()

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

    def print_help(self):
        print(self.help_text)
