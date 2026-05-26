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
from modules.util.image_util import load_image
from modules.util.torch_util import default_device, torch_gc
from modules.util.ui import components
from modules.util.ui.ui_utils import bind_mousewheel, set_window_icon
from modules.util.ui.UIState import UIState

import torch

import customtkinter as ctk
import cv2
import numpy as np
from customtkinter import ScalingTracker, ThemeManager
from PIL import Image, ImageDraw


class CaptionUI(ctk.CTkToplevel):
    def __init__(
            self,
            parent,
            initial_dir: str | None,
            initial_include_subdirectories: bool,
            *args,
            **kwargs,
    ) -> None:
        super().__init__(parent, *args, **kwargs)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.dir = initial_dir
        self.config_ui_data = {"include_subdirectories": initial_include_subdirectories}
        self.config_ui_state = UIState(self, self.config_ui_data)
        self.image_size = 850
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
        self.image_rel_paths = []
        self.current_image_index = -1
        self.file_list = None
        self.image_labels = []
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
        self.mask_editing_mode = 'draw'
        self.enable_mask_editing_var = ctk.BooleanVar()
        self.mask_editing_alpha = None
        self.prompt_var = None
        self.prompt_component = None


        self.title("OneTrainer")
        self.geometry("1280x980")
        self.resizable(False, False)


        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)


        self.top_bar(self)

        self.bottom_frame = ctk.CTkFrame(self)
        self.bottom_frame.grid(row=1, column=0, sticky="nsew")
        self.bottom_frame.grid_rowconfigure(0, weight=1)
        self.bottom_frame.grid_columnconfigure(0, weight=0)
        self.bottom_frame.grid_columnconfigure(1, weight=1)

        self.file_list_column(self.bottom_frame)
        self.content_column(self.bottom_frame)
        self.load_directory()

        self.wait_visibility()
        self.focus_set()
        self.after(200, lambda: set_window_icon(self))

    def top_bar(self, master):
        top_frame = ctk.CTkFrame(master)
        top_frame.grid(row=0, column=0, sticky="nsew")

        components.button(top_frame, 0, 0, "Open", self.open_directory,
                          tooltip="open a new directory")
        components.button(top_frame, 0, 1, "Generate Masks", self.open_mask_window,
                          tooltip="open a dialog to automatically generate masks")
        components.button(top_frame, 0, 2, "Generate Captions", self.open_caption_window,
                          tooltip="open a dialog to automatically generate captions")

        if platform.system() == "Windows":
            components.button(top_frame, 0, 3, "Open in Explorer", self.open_in_explorer,
                              tooltip="open the current image in Explorer")

        components.switch(top_frame, 0, 4, self.config_ui_state, "include_subdirectories",
                          text="include subdirectories")

        top_frame.grid_columnconfigure(5, weight=1)

        components.button(top_frame, 0, 6, "Help", self.print_help,
                          tooltip=self.help_text)

    def file_list_column(self, master):
        if self.file_list is not None:
            self.image_labels = []
            self.file_list.destroy()

        self.file_list = ctk.CTkScrollableFrame(master, width=300)
        self.file_list.grid(row=0, column=0, sticky="nsew")

        for i, filename in enumerate(self.image_rel_paths):
            def __create_switch_image(index):
                def __switch_image(event):
                    self.switch_image(index)

                return __switch_image

            label = ctk.CTkLabel(self.file_list, text=filename)
            label.bind("<Button-1>", __create_switch_image(i))

            self.image_labels.append(label)
            label.grid(row=i, column=0, padx=5, sticky="nsw")

    def content_column(self, master):
        image = Image.new("RGBA", (512, 512), (0, 0, 0, 0))

        right_frame = ctk.CTkFrame(master, fg_color="transparent")
        right_frame.grid(row=0, column=1, sticky="nsew")

        right_frame.grid_columnconfigure(4, weight=1)
        right_frame.grid_rowconfigure(1, weight=1)

        components.button(right_frame, 0, 0, "Draw", self.draw_mask_editing_mode,
                          tooltip="draw a mask using a brush")
        components.button(right_frame, 0, 1, "Fill", self.fill_mask_editing_mode,
                          tooltip="draw a mask using a fill tool")

        # checkbox to enable mask editing
        self.enable_mask_editing_var = ctk.BooleanVar()
        self.enable_mask_editing_var.set(False)
        enable_mask_editing_checkbox = ctk.CTkCheckBox(
            right_frame, text="Enable Mask Editing", variable=self.enable_mask_editing_var, width=50)
        enable_mask_editing_checkbox.grid(row=0, column=2, padx=25, pady=5, sticky="w")

        # mask alpha textbox
        self.mask_editing_alpha = ctk.CTkEntry(master=right_frame, width=40, placeholder_text="1.0")
        self.mask_editing_alpha.insert(0, "1.0")
        self.mask_editing_alpha.grid(row=0, column=3, sticky="e", padx=5, pady=5)
        self.bind_key_events(self.mask_editing_alpha)

        mask_editing_alpha_label = ctk.CTkLabel(right_frame, text="Brush Alpha", width=75)
        mask_editing_alpha_label.grid(row=0, column=4, padx=0, pady=5, sticky="w")

        # image
        self.image = ctk.CTkImage(
            light_image=image,
            size=(self.image_size, self.image_size)
        )
        self.image_label = ctk.CTkLabel(
            master=right_frame, text="", image=self.image, height=self.image_size, width=self.image_size
        )
        self.image_label.grid(row=1, column=0, columnspan=5, sticky="nsew")

        self.image_label.bind("<Motion>", self.edit_mask)
        self.image_label.bind("<Button-1>", self.edit_mask)
        self.image_label.bind("<Button-3>", self.edit_mask)
        bind_mousewheel(self.image_label, {self.image_label.children["!label"]}, self.draw_mask_radius)

        # prompt
        self.prompt_var = ctk.StringVar()
        self.prompt_component = ctk.CTkEntry(right_frame, textvariable=self.prompt_var)
        self.prompt_component.grid(row=2, column=0, columnspan=5, pady=5, sticky="new")
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
            return path_util.is_supported_image_extension(ext) and not name.endswith("-masklabel") and not name.endswith("-condlabel")

        self.image_rel_paths = []

        if not self.dir or not os.path.isdir(self.dir):
            return

        if include_subdirectories:
            for root, _, files in os.walk(self.dir):
                for filename in files:
                    if __is_supported_image_extension(filename):
                        self.image_rel_paths.append(
                            os.path.relpath(os.path.join(root, filename), self.dir)
                        )
        else:
            for _, filename in enumerate(os.listdir(self.dir)):
                if __is_supported_image_extension(filename):
                    self.image_rel_paths.append(
                        os.path.relpath(os.path.join(self.dir, filename), self.dir)
                    )

    def load_image(self):
        image_name = "resources/icons/icon.png"

        if len(self.image_rel_paths) > 0 and self.current_image_index < len(self.image_rel_paths):
            image_name = self.image_rel_paths[self.current_image_index]
            image_name = os.path.join(self.dir, image_name)

        try:
            return load_image(image_name, convert_mode="RGB")
        except Exception:
            print(f'Could not open image {image_name}')

    def load_mask(self):
        if len(self.image_rel_paths) > 0 and self.current_image_index < len(self.image_rel_paths):
            image_name = self.image_rel_paths[self.current_image_index]
            mask_name = os.path.splitext(image_name)[0] + "-masklabel.png"
            mask_name = os.path.join(self.dir, mask_name)

            try:
                return load_image(mask_name, convert_mode='RGB')
            except Exception:
                return None
        else:
            return None

    def load_prompt(self):
        if len(self.image_rel_paths) > 0 and self.current_image_index < len(self.image_rel_paths):
            image_name = self.image_rel_paths[self.current_image_index]
            prompt_name = os.path.splitext(image_name)[0] + ".txt"
            prompt_name = os.path.join(self.dir, prompt_name)

            try:
                with open(prompt_name, "r", encoding='utf-8') as f:
                    return f.readlines()[0].strip()
            except Exception:
                return ""
        else:
            return ""

    def previous_image(self, event):
        if len(self.image_rel_paths) > 0 and (self.current_image_index - 1) >= 0:
            self.switch_image(self.current_image_index - 1)

    def next_image(self, event):
        if len(self.image_rel_paths) > 0 and (self.current_image_index + 1) < len(self.image_rel_paths):
            self.switch_image(self.current_image_index + 1)

    def switch_image(self, index):
        if len(self.image_labels) > 0 and self.current_image_index < len(self.image_labels):
            self.image_labels[self.current_image_index].configure(
                text_color=ThemeManager.theme["CTkLabel"]["text_color"])

        self.current_image_index = index
        if index >= 0:
            self.image_labels[index].configure(text_color="#FF0000")

            self.pil_image = self.load_image()
            self.pil_mask = self.load_mask()
            prompt = self.load_prompt()

            self.image_width = self.pil_image.width
            self.image_height = self.pil_image.height
            scale = self.image_size / max(self.pil_image.height, self.pil_image.width)
            height = int(self.pil_image.height * scale)
            width = int(self.pil_image.width * scale)

            self.pil_image = self.pil_image.resize((width, height), Image.Resampling.LANCZOS)

            self.refresh_image()
            self.prompt_var.set(prompt)
        else:
            image = Image.new("RGB", (512, 512), (0, 0, 0))
            self.image.configure(light_image=image)

    def refresh_image(self):
        if self.pil_mask:
            resized_pil_mask = self.pil_mask.resize(
                (self.pil_image.width, self.pil_image.height),
                Image.Resampling.NEAREST
            )

            if self.display_only_mask:
                self.image.configure(light_image=resized_pil_mask, size=resized_pil_mask.size)
            else:
                np_image = np.array(self.pil_image).astype(np.float32) / 255.0
                np_mask = np.array(resized_pil_mask).astype(np.float32) / 255.0

                # normalize mask between 0.3 - 1.0 so we can see image underneath and gauge strength of the alpha
                norm_min = 0.3
                np_mask_min = np_mask.min()
                if np_mask_min == 0:
                    # optimize for common case
                    np_mask = np_mask * (1.0 - norm_min) + norm_min
                elif np_mask_min < 1:
                    # note: min of 1 means we get divide by 0
                    np_mask = (np_mask - np_mask_min) / (1.0 - np_mask_min) * (1.0 - norm_min) + norm_min

                np_masked_image = (np_image * np_mask * 255.0).astype(np.uint8)
                masked_image = Image.fromarray(np_masked_image, mode='RGB')

                self.image.configure(light_image=masked_image, size=masked_image.size)
        else:
            self.image.configure(light_image=self.pil_image, size=self.pil_image.size)

    def draw_mask_radius(self, delta, raw_event):
        # Wheel up = Increase radius. Wheel down = Decrease radius.
        multiplier = 1.0 + (delta * 0.05)
        self.mask_draw_radius = max(0.0025, self.mask_draw_radius * multiplier)

    def edit_mask(self, event):
        if not self.enable_mask_editing_var.get():
            return

        if event.widget != self.image_label.children["!label"]:
            return

        if len(self.image_rel_paths) == 0 or self.current_image_index >= len(self.image_rel_paths):
            return

        display_scaling = ScalingTracker.get_window_scaling(self)

        event_x = event.x / display_scaling
        event_y = event.y / display_scaling

        start_x = int(event_x / self.pil_image.width * self.image_width)
        start_y = int(event_y / self.pil_image.height * self.image_height)
        end_x = int(self.mask_draw_x / self.pil_image.width * self.image_width)
        end_y = int(self.mask_draw_y / self.pil_image.height * self.image_height)

        self.mask_draw_x = event_x
        self.mask_draw_y = event_y

        is_right = False
        is_left = False
        if event.state & 0x0100 or event.num == 1:  # left mouse button
            is_left = True
        elif event.state & 0x0400 or event.num == 3:  # right mouse button
            is_right = True

        if self.mask_editing_mode == 'draw':
            self.draw_mask(start_x, start_y, end_x, end_y, is_left, is_right)
        if self.mask_editing_mode == 'fill':
            self.fill_mask(start_x, start_y, end_x, end_y, is_left, is_right)

    def draw_mask(self, start_x, start_y, end_x, end_y, is_left, is_right):
        color = None

        adding_to_mask = True
        if is_left:
            try:
                alpha = float(self.mask_editing_alpha.get())
            except Exception:
                alpha = 1.0
            rgb_value = int(max(0.0, min(alpha, 1.0)) * 255)  # max/min stuff to clamp to 0 - 255 range
            color = (rgb_value, rgb_value, rgb_value)

        elif is_right:
            color = (0, 0, 0)
            adding_to_mask = False

        if color is not None:
            if self.pil_mask is None:
                if adding_to_mask:
                    self.pil_mask = Image.new('RGB', size=(self.image_width, self.image_height), color=(0, 0, 0))
                else:
                    self.pil_mask = Image.new('RGB', size=(self.image_width, self.image_height), color=(255, 255, 255))

            radius = int(self.mask_draw_radius * max(self.pil_mask.width, self.pil_mask.height))

            draw = ImageDraw.Draw(self.pil_mask)
            draw.line((start_x, start_y, end_x, end_y), fill=color,
                      width=radius + radius + 1)
            draw.ellipse((start_x - radius, start_y - radius,
                          start_x + radius, start_y + radius), fill=color, outline=None)
            draw.ellipse((end_x - radius, end_y - radius, end_x + radius,
                          end_y + radius), fill=color, outline=None)

            self.refresh_image()

    def fill_mask(self, start_x, start_y, end_x, end_y, is_left, is_right):
        color = None

        adding_to_mask = True
        if is_left:
            try:
                alpha = float(self.mask_editing_alpha.get())
            except Exception:
                alpha = 1.0
            rgb_value = int(max(0.0, min(alpha, 1.0)) * 255)  # max/min stuff to clamp to 0 - 255 range
            color = (rgb_value, rgb_value, rgb_value)

        elif is_right:
            color = (0, 0, 0)
            adding_to_mask = False

        if color is not None:
            if self.pil_mask is None:
                if adding_to_mask:
                    self.pil_mask = Image.new('RGB', size=(self.image_width, self.image_height), color=(0, 0, 0))
                else:
                    self.pil_mask = Image.new('RGB', size=(self.image_width, self.image_height), color=(255, 255, 255))

            np_mask = np.array(self.pil_mask).astype(np.uint8)
            cv2.floodFill(np_mask, None, (start_x, start_y), color)
            self.pil_mask = Image.fromarray(np_mask, 'RGB')

            self.refresh_image()

    def save(self, event):
        if len(self.image_rel_paths) > 0 and self.current_image_index < len(self.image_rel_paths):
            image_name = self.image_rel_paths[self.current_image_index]

            prompt_name = os.path.splitext(image_name)[0] + ".txt"
            prompt_name = os.path.join(self.dir, prompt_name)

            mask_name = os.path.splitext(image_name)[0] + "-masklabel.png"
            mask_name = os.path.join(self.dir, mask_name)

            try:
                with open(prompt_name, "w", encoding='utf-8') as f:
                    f.write(self.prompt_var.get())
            except Exception:
                return

            if self.pil_mask:
                self.pil_mask.save(mask_name)

    def draw_mask_editing_mode(self, *args):
        self.mask_editing_mode = 'draw'

        if args:
            # disable default event
            return "break"
        return None

    def fill_mask_editing_mode(self, *args):
        self.mask_editing_mode = 'fill'

    def toggle_mask(self, *args):
        self.display_only_mask = not self.display_only_mask
        self.refresh_image()

    def open_directory(self):
        new_dir = filedialog.askdirectory()

        if new_dir:
            self.dir = new_dir
            self.load_directory(include_subdirectories=self.config_ui_data["include_subdirectories"])

    def open_mask_window(self):
        dialog = GenerateMasksWindow(self, self.dir, self.config_ui_data["include_subdirectories"])
        self.wait_window(dialog)
        self.switch_image(self.current_image_index)

    def open_caption_window(self):
        dialog = GenerateCaptionsWindow(self, self.dir, self.config_ui_data["include_subdirectories"])
        self.wait_window(dialog)
        self.switch_image(self.current_image_index)

    def open_in_explorer(self):
        try:
            image_name = self.image_rel_paths[self.current_image_index]
            image_name = os.path.realpath(os.path.join(self.dir, image_name))
            subprocess.Popen(f"explorer /select,{image_name}")
        except Exception:
            traceback.print_exc()

    def load_masking_model(self, model):
        model_type = type(self.masking_model).__name__ if self.masking_model else None

        if model == "ClipSeg" and model_type != "ClipSegModel":
            self._release_models()
            print("loading ClipSeg model, this may take a while")
            self.masking_model = ClipSegModel(default_device, torch.float32)
        elif model == "Rembg" and model_type != "RembgModel":
            self._release_models()
            print("loading Rembg model, this may take a while")
            self.masking_model = RembgModel(default_device, torch.float32)
        elif model == "Rembg-Human" and model_type != "RembgHumanModel":
            self._release_models()
            print("loading Rembg-Human model, this may take a while")
            self.masking_model = RembgHumanModel(default_device, torch.float32)
        elif model == "Hex Color" and model_type != "MaskByColor":
            self._release_models()
            self.masking_model = MaskByColor(default_device, torch.float32)

    def load_captioning_model(self, model):
        model_type = type(self.captioning_model).__name__ if self.captioning_model else None

        if model == "Blip" and model_type != "BlipModel":
            self._release_models()
            print("loading Blip model, this may take a while")
            self.captioning_model = BlipModel(default_device, torch.float16)
        elif model == "Blip2" and model_type != "Blip2Model":
            self._release_models()
            print("loading Blip2 model, this may take a while")
            self.captioning_model = Blip2Model(default_device, torch.float16)
        elif model == "WD14 VIT v2" and model_type != "WDModel":
            self._release_models()
            print("loading WD14_VIT_v2 model, this may take a while")
            self.captioning_model = WDModel(default_device, torch.float16)

    def print_help(self):
        print(self.help_text)

    def _release_models(self):
        """Release all models from VRAM"""
        freed = False
        if self.captioning_model is not None:
            self.captioning_model = None
            freed = True
        if self.masking_model is not None:
            self.masking_model = None
            freed = True
        if freed:
            torch_gc()

    def _on_close(self):
        self._release_models()
        self.destroy()

    def destroy(self):
        self._release_models()
        super().destroy()
