import os
import platform
import subprocess
import traceback
from tkinter import filedialog

import customtkinter as ctk
import numpy as np
import torch
from PIL import Image, ImageDraw
from customtkinter import ThemeManager

from modules.module.Blip2Model import Blip2Model
from modules.module.BlipModel import BlipModel
from modules.module.ClipSegModel import ClipSegModel
from modules.module.RembgModel import RembgModel
from modules.ui.GenerateCaptionsWindow import GenerateCaptionsWindow
from modules.ui.GenerateMasksWindow import GenerateMasksWindow
from modules.util import path_util
from modules.util.ui import components


class CaptionUI(ctk.CTkToplevel):
    def __init__(self, parent, initial_dir: str | None, *args, **kwargs):
        ctk.CTkToplevel.__init__(self, parent, *args, **kwargs)

        self.dir = initial_dir
        self.image_size = 900

        self.title("OneTrainer")
        self.geometry("1280x1024")
        self.resizable(False, False)
        self.wait_visibility()
        self.grab_set()
        self.focus_set()

        self.help_text = """
Keyboard shortcuts when focusing on the prompt input field:
Up arrow: previous image
Down arrow: next image
Return: save
Ctrl+M: only show the mask

When editing masks:
Left click: add mask
Right click: remove mask
Mouse wheel: increase or decrease brush size"""

        self.masking_model = None
        self.captioning_model = None

        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.image_names = []
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
        self.enable_mask_editing_var = ctk.BooleanVar()
        self.prompt_var = None
        self.prompt_component = None
        self.content_column(self.bottom_frame)

        self.load_directory()

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

        top_frame.grid_columnconfigure(4, weight=1)

        components.button(top_frame, 0, 5, "Help", self.print_help,
                          tooltip=self.help_text)

    def file_list_column(self, master):
        if self.file_list is not None:
            self.image_labels = []
            self.file_list.destroy()

        self.file_list = ctk.CTkScrollableFrame(master, width=300)
        self.file_list.grid(row=0, column=0, sticky="nsew")

        for i, filename in enumerate(self.image_names):
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

        right_frame.grid_columnconfigure(0, weight=1)
        right_frame.grid_rowconfigure(0, weight=1)

        # checkbox to enable mask editing
        self.enable_mask_editing_var = ctk.BooleanVar()
        self.enable_mask_editing_var.set(False)
        enable_mask_editing_checkbox = ctk.CTkCheckBox(
            right_frame, text="Enable Mask Editing", variable=self.enable_mask_editing_var, width=50)
        enable_mask_editing_checkbox.grid(row=0, column=0, padx=10, pady=5, sticky="nsew")

        # image
        self.image = ctk.CTkImage(
            light_image=image,
            size=(self.image_size, self.image_size)
        )
        self.image_label = ctk.CTkLabel(
            master=right_frame, text="", image=self.image, height=self.image_size, width=self.image_size
        )
        self.image_label.grid(row=1, column=0, sticky="nsew")

        self.image_label.bind("<Motion>", self.draw_mask)
        self.image_label.bind("<Button-1>", self.draw_mask)
        self.image_label.bind("<Button-3>", self.draw_mask)
        self.image_label.bind("<MouseWheel>", self.draw_mask_radius)

        # prompt
        self.prompt_var = ctk.StringVar()
        self.prompt_component = ctk.CTkEntry(right_frame, textvariable=self.prompt_var)
        self.prompt_component.grid(row=2, column=0, pady=5, sticky="new")
        self.prompt_component.bind("<Down>", self.next_image)
        self.prompt_component.bind("<Up>", self.previous_image)
        self.prompt_component.bind("<Return>", self.save)
        self.prompt_component.bind("<Control-m>", self.toggle_mask)
        self.prompt_component.focus_set()

    def load_directory(self):
        self.scan_directory()
        self.file_list_column(self.bottom_frame)

        if len(self.image_names) > 0:
            self.switch_image(0)
        else:
            self.switch_image(-1)

        self.prompt_component.focus_set()

    def scan_directory(self):
        self.image_names = []

        if self.dir and os.path.isdir(self.dir):
            for i, filename in enumerate(os.listdir(self.dir)):
                path = path_util.canonical_join(self.dir, filename)
                name, ext = os.path.splitext(path)
                if path_util.is_supported_image_extension(ext) and not name.endswith("-masklabel"):
                    self.image_names.append(filename)

    def load_image(self):
        image_name = "resources/icons/icon.png"

        if len(self.image_names) > 0 and self.current_image_index < len(self.image_names):
            image_name = self.image_names[self.current_image_index]
            image_name = os.path.join(self.dir, image_name)

        try:
            return Image.open(image_name).convert('RGB')
        except:
            print(f'Could not open image {image_name}')

    def load_mask(self):
        if len(self.image_names) > 0 and self.current_image_index < len(self.image_names):
            image_name = self.image_names[self.current_image_index]
            mask_name = os.path.splitext(image_name)[0] + "-masklabel.png"
            mask_name = os.path.join(self.dir, mask_name)

            try:
                return Image.open(mask_name).convert('RGB')
            except:
                return None
        else:
            return None

    def load_prompt(self):
        if len(self.image_names) > 0 and self.current_image_index < len(self.image_names):
            image_name = self.image_names[self.current_image_index]
            prompt_name = os.path.splitext(image_name)[0] + ".txt"
            prompt_name = os.path.join(self.dir, prompt_name)

            try:
                with open(prompt_name, "r") as f:
                    return f.readlines()[0]
            except:
                return ""
        else:
            return ""

    def previous_image(self, event):
        if len(self.image_names) > 0 and (self.current_image_index - 1) >= 0:
            self.switch_image(self.current_image_index - 1)

    def next_image(self, event):
        if len(self.image_names) > 0 and (self.current_image_index + 1) < len(self.image_names):
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
                np_mask = np.clip(np_mask, 0.4, 1.0)
                np_masked_image = (np_image * np_mask * 255.0).astype(np.uint8)
                masked_image = Image.fromarray(np_masked_image, mode='RGB')

                self.image.configure(light_image=masked_image, size=masked_image.size)
        else:
            self.image.configure(light_image=self.pil_image, size=self.pil_image.size)

    def draw_mask_radius(self, event):
        if event.widget != self.image_label.children["!label"]:
            return

        delta = 1.0 + (-np.sign(event.delta) * 0.05)
        self.mask_draw_radius *= delta

    def draw_mask(self, event):
        if not self.enable_mask_editing_var.get():
            return

        if event.widget != self.image_label.children["!label"]:
            return

        if len(self.image_names) == 0 or self.current_image_index >= len(self.image_names):
            return

        start_x = int(event.x / self.pil_image.width * self.image_width)
        start_y = int(event.y / self.pil_image.height * self.image_height)
        end_x = int(self.mask_draw_x / self.pil_image.width * self.image_width)
        end_y = int(self.mask_draw_y / self.pil_image.height * self.image_height)

        self.mask_draw_x = event.x
        self.mask_draw_y = event.y

        color = None

        if event.state & 0x0100 or event.num == 1:  # left mouse button
            color = (255, 255, 255)
        elif event.state & 0x0400 or event.num == 3:  # right mouse button
            color = (0, 0, 0)

        if color is not None:
            if self.pil_mask is None:
                self.pil_mask = Image.new('RGB', size=(self.image_width, self.image_height), color=(0, 0, 0))

            radius = int(self.mask_draw_radius * max(self.pil_mask.width, self.pil_mask.height))

            draw = ImageDraw.Draw(self.pil_mask)
            draw.line((start_x, start_y, end_x, end_y), fill=color,
                      width=radius + radius + 1)
            draw.ellipse((start_x - radius, start_y - radius,
                          start_x + radius, start_y + radius), fill=color, outline=None)
            draw.ellipse((end_x - radius, end_y - radius, end_x + radius,
                          end_y + radius), fill=color, outline=None)

            self.refresh_image()

    def save(self, event):
        if len(self.image_names) > 0 and self.current_image_index < len(self.image_names):
            image_name = self.image_names[self.current_image_index]

            prompt_name = os.path.splitext(image_name)[0] + ".txt"
            prompt_name = os.path.join(self.dir, prompt_name)

            mask_name = os.path.splitext(image_name)[0] + "-masklabel.png"
            mask_name = os.path.join(self.dir, mask_name)

            try:
                with open(prompt_name, "w") as f:
                    f.write(self.prompt_var.get())
            except:
                return ""

            if self.pil_mask:
                self.pil_mask.save(mask_name)

        else:
            return ""

    def toggle_mask(self, event):
        self.display_only_mask = not self.display_only_mask
        self.refresh_image()

    def open_directory(self):
        new_dir = filedialog.askdirectory()

        if new_dir:
            self.dir = new_dir
            self.load_directory()

    def open_mask_window(self):
        dialog = GenerateMasksWindow(self, self.dir)
        self.wait_window(dialog)
        self.switch_image(self.current_image_index)

    def open_caption_window(self):
        dialog = GenerateCaptionsWindow(self, self.dir)
        self.wait_window(dialog)
        self.switch_image(self.current_image_index)

    def open_in_explorer(self):
        try:
            image_name = self.image_names[self.current_image_index]
            image_name = os.path.realpath(os.path.join(self.dir, image_name))
            subprocess.Popen(f"explorer /select,{image_name}")
        except:
            traceback.print_exc()
            pass

    def load_masking_model(self, model):
        self.captioning_model = None

        if model == "ClipSeg":
            if self.masking_model is None or not isinstance(self.masking_model, ClipSegModel):
                print("loading ClipSeg model, this may take a while")
                self.masking_model = ClipSegModel(torch.device("cuda"), torch.float32)
        elif model == "Rembg":
            if self.masking_model is None or not isinstance(self.masking_model, RembgModel):
                print("loading Rembg model, this may take a while")
                self.masking_model = RembgModel(torch.device("cuda"), torch.float32)

    def load_captioning_model(self, model):
        self.masking_model = None

        if model == "Blip":
            if self.captioning_model is None or not isinstance(self.captioning_model, BlipModel):
                print("loading Blip model, this may take a while")
                self.captioning_model = BlipModel(torch.device("cuda"), torch.float16)
        elif model == "Blip2":
            if self.captioning_model is None or not isinstance(self.captioning_model, Blip2Model):
                print("loading Blip2 model, this may take a while")
                self.captioning_model = Blip2Model(torch.device("cuda"), torch.float16)

    def print_help(self):
        print(self.help_text)
