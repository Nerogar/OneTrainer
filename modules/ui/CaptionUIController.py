import os
import subprocess
import traceback

from modules.module.Blip2Model import Blip2Model
from modules.module.BlipModel import BlipModel
from modules.module.ClipSegModel import ClipSegModel
from modules.module.MaskByColor import MaskByColor
from modules.module.RembgHumanModel import RembgHumanModel
from modules.module.RembgModel import RembgModel
from modules.module.WDModel import WDModel
from modules.ui.GenerateCaptionsWindowController import GenerateCaptionsWindowController
from modules.ui.GenerateMasksWindowController import GenerateMasksWindowController
from modules.util import path_util
from modules.util.image_util import load_image
from modules.util.torch_util import default_device, torch_gc

import torch

import numpy as np
from PIL import Image, ImageDraw


class CaptionUIController:
    def __init__(self, initial_dir: str | None, initial_include_subdirectories: bool):
        self.dir = initial_dir
        self.config_ui_data = {"include_subdirectories": initial_include_subdirectories}
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
        self.pil_image = None
        self.image_width = 0
        self.image_height = 0
        self.pil_mask = None
        self.mask_draw_x = 0
        self.mask_draw_y = 0
        self.mask_draw_radius = 0.01
        self.display_only_mask = False
        self.mask_editing_mode = 'draw'
        self.view = None

    def create_window(self, parent, view_cls):
        self.view = view_cls(parent, self)
        return self.view

    def open_mask_window(self, parent_window, view_cls):
        controller = GenerateMasksWindowController(self)
        return controller.create_window(parent_window, self.dir, self.config_ui_data["include_subdirectories"], view_cls)

    def open_caption_window(self, parent_window, view_cls):
        controller = GenerateCaptionsWindowController(self)
        return controller.create_window(parent_window, self.dir, self.config_ui_data["include_subdirectories"], view_cls)

    def open_in_explorer(self):
        try:
            image_name = self.image_rel_paths[self.current_image_index]
            image_name = os.path.realpath(os.path.join(self.dir, image_name))
            subprocess.Popen(f"explorer /select,{image_name}")
        except Exception:
            traceback.print_exc()

    def switch_image(self, index):
        old_index = self.current_image_index
        self.current_image_index = index
        if index >= 0:
            self.pil_image = self.load_image()
            self.pil_mask = self.load_mask()
            prompt = self.load_prompt()

            self.image_width = self.pil_image.width
            self.image_height = self.pil_image.height
            scale = self.image_size / max(self.pil_image.height, self.pil_image.width)
            height = int(self.pil_image.height * scale)
            width = int(self.pil_image.width * scale)

            self.pil_image = self.pil_image.resize((width, height), Image.Resampling.LANCZOS)

            self.view.on_image_switched(old_index, index, prompt)
        else:
            self.view.on_image_cleared()

    def previous_image(self):
        if len(self.image_rel_paths) > 0 and (self.current_image_index - 1) >= 0:
            self.switch_image(self.current_image_index - 1)

    def next_image(self):
        if len(self.image_rel_paths) > 0 and (self.current_image_index + 1) < len(self.image_rel_paths):
            self.switch_image(self.current_image_index + 1)

    def load_directory(self, include_subdirectories: bool = False):
        self.scan_directory(include_subdirectories)
        self.view.refresh_file_list()

        if len(self.image_rel_paths) > 0:
            self.switch_image(0)
        else:
            self.switch_image(-1)

        self.view.focus_prompt()

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

    def save(self, prompt_text):
        if len(self.image_rel_paths) > 0 and self.current_image_index < len(self.image_rel_paths):
            image_name = self.image_rel_paths[self.current_image_index]

            prompt_name = os.path.splitext(image_name)[0] + ".txt"
            prompt_name = os.path.join(self.dir, prompt_name)

            mask_name = os.path.splitext(image_name)[0] + "-masklabel.png"
            mask_name = os.path.join(self.dir, mask_name)

            try:
                with open(prompt_name, "w", encoding='utf-8') as f:
                    f.write(prompt_text)
            except Exception:
                return

            if self.pil_mask:
                self.pil_mask.save(mask_name)

    def get_display_image(self):
        if self.pil_mask:
            resized_pil_mask = self.pil_mask.resize(
                (self.pil_image.width, self.pil_image.height),
                Image.Resampling.NEAREST
            )

            if self.display_only_mask:
                return resized_pil_mask, resized_pil_mask.size
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

                return masked_image, masked_image.size
        else:
            return self.pil_image, self.pil_image.size

    def toggle_mask(self):
        self.display_only_mask = not self.display_only_mask

    def set_mask_editing_mode(self, mode):
        self.mask_editing_mode = mode

    def update_mask_draw_radius(self, delta):
        # Wheel up = Increase radius. Wheel down = Decrease radius.
        multiplier = 1.0 + (delta * 0.05)
        self.mask_draw_radius = max(0.0025, self.mask_draw_radius * multiplier)

    def handle_edit_mask(self, event_x, event_y, is_left, is_right, alpha):
        if len(self.image_rel_paths) == 0 or self.current_image_index >= len(self.image_rel_paths):
            return
        if self.pil_image is None:
            return

        start_x = int(event_x / self.pil_image.width * self.image_width)
        start_y = int(event_y / self.pil_image.height * self.image_height)
        end_x = int(self.mask_draw_x / self.pil_image.width * self.image_width)
        end_y = int(self.mask_draw_y / self.pil_image.height * self.image_height)

        self.mask_draw_x = event_x
        self.mask_draw_y = event_y

        if self.mask_editing_mode == 'draw':
            self.draw_mask(start_x, start_y, end_x, end_y, is_left, is_right, alpha)
        if self.mask_editing_mode == 'fill':
            self.fill_mask(start_x, start_y, end_x, end_y, is_left, is_right, alpha)

    def draw_mask(self, start_x, start_y, end_x, end_y, is_left, is_right, alpha):
        color = None

        adding_to_mask = True
        if is_left:
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

            self.view.refresh_image()

    def fill_mask(self, start_x, start_y, end_x, end_y, is_left, is_right, alpha):
        color = None

        adding_to_mask = True
        if is_left:
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
            import cv2
            cv2.floodFill(np_mask, None, (start_x, start_y), color)
            self.pil_mask = Image.fromarray(np_mask, 'RGB')

            self.view.refresh_image()

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

    def on_close(self):
        self._release_models()
        self.view.destroy()
