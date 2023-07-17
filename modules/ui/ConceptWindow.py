import os
import random

import customtkinter as ctk
from PIL import Image
from torchvision.transforms import functional, InterpolationMode

from modules.util import path_util
from modules.util.ui import components
from modules.util.ui.UIState import UIState


class ConceptWindow(ctk.CTkToplevel):
    def __init__(self, parent, concept, *args, **kwargs):
        ctk.CTkToplevel.__init__(self, parent, *args, **kwargs)

        self.ui_state = UIState(self, concept)

        self.concept = concept

        self.title("Concept")
        self.geometry("800x450")
        self.resizable(False, False)
        self.wait_visibility()
        self.grab_set()
        self.focus_set()

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        tabview = ctk.CTkTabview(self)
        tabview.grid(row=0, column=0, sticky="nsew")

        self.__general_tab(tabview.add("general"), concept)
        self.__augmentation_tab(tabview.add("augmentation"))

        components.button(self, 1, 0, "ok", self.__ok)

    def __general_tab(self, master, concept):
        master.grid_columnconfigure(1, weight=1)
        master.grid_columnconfigure(2, weight=1)

        # name
        components.label(master, 0, 0, "Name",
                         tooltip="Name of the concept")
        components.entry(master, 0, 1, self.ui_state, "name")

        # path
        components.label(master, 1, 0, "Path",
                         tooltip="Path where the training data is located")
        components.dir_entry(master, 1, 1, self.ui_state, "path")

        # prompt mode
        components.label(master, 2, 0, "Prompt Source",
                         tooltip="The source for prompts used during training. When selecting \"From single text file\", select a text file that contains a list of prompts")
        prompt_path_entry = components.file_entry(master, 2, 2, self.ui_state, "prompt_path")

        def set_prompt_path_entry_enabled(option: str):
            if option == 'concept':
                for child in prompt_path_entry.children.values():
                    child.configure(state="normal")
            else:
                for child in prompt_path_entry.children.values():
                    child.configure(state="disabled")

        components.options_kv(master, 2, 1, [
            ("From text file per sample", 'sample'),
            ("From single text file", 'concept'),
            ("From image file name", 'filename'),
        ], self.ui_state, "prompt_source", command=set_prompt_path_entry_enabled)
        set_prompt_path_entry_enabled(concept["prompt_source"])

        # include subdirectories
        components.label(master, 3, 0, "Include Subdirectories",
                         tooltip="Includes images from subdirectories into the dataset")
        components.switch(master, 3, 1, self.ui_state, "include_subdirectories")

    def __augmentation_tab(self, master):
        master.grid_columnconfigure(0, weight=0)
        master.grid_columnconfigure(1, weight=0)
        master.grid_columnconfigure(2, weight=0)
        master.grid_columnconfigure(3, weight=1)

        # crop jitter
        components.label(master, 0, 0, "Crop Jitter",
                         tooltip="Enables random cropping of samples")
        components.switch(master, 0, 1, self.ui_state, "enable_crop_jitter")

        # random flip
        components.label(master, 1, 0, "Random Flip",
                         tooltip="Randomly flip the sample during training")
        components.switch(master, 1, 1, self.ui_state, "enable_random_flip")

        # random rotation
        components.label(master, 2, 0, "Random Rotation",
                         tooltip="Randomly rotates the sample during training")
        components.switch(master, 2, 1, self.ui_state, "enable_random_rotate")
        components.entry(master, 2, 2, self.ui_state, "random_rotate_max_angle")

        # random brightness
        components.label(master, 3, 0, "Random Brightness",
                         tooltip="Randomly adjusts the brightness of the sample during training")
        components.switch(master, 3, 1, self.ui_state, "enable_random_brightness")
        components.entry(master, 3, 2, self.ui_state, "random_brightness_max_strength")

        # random contrast
        components.label(master, 4, 0, "Random Contrast",
                         tooltip="Randomly adjusts the contrast of the sample during training")
        components.switch(master, 4, 1, self.ui_state, "enable_random_contrast")
        components.entry(master, 4, 2, self.ui_state, "random_contrast_max_strength")

        # random saturation
        components.label(master, 5, 0, "Random Saturation",
                         tooltip="Randomly adjusts the saturation of the sample during training")
        components.switch(master, 5, 1, self.ui_state, "enable_random_saturation")
        components.entry(master, 5, 2, self.ui_state, "random_saturation_max_strength")

        # random hue
        components.label(master, 6, 0, "Random Hue",
                         tooltip="Randomly adjusts the hue of the sample during training")
        components.switch(master, 6, 1, self.ui_state, "enable_random_hue")
        components.entry(master, 6, 2, self.ui_state, "random_hue_max_strength")

        # image
        self.image = ctk.CTkImage(
            light_image=self.__get_preview_image(),
            size=(300, 300)
        )
        image_label = ctk.CTkLabel(master=master, text="", image=self.image, height=150, width=150)
        image_label.grid(row=0, column=3, rowspan=6)

        # refresh preview
        components.button(master, 6, 3, "Update Preview", command=self.__update_preview)

    def __update_preview(self):
        self.image.configure(light_image=self.__get_preview_image())

    def __get_preview_image(self):
        preview_path = "resources/icons/icon.png"

        if os.path.isdir(self.concept["path"]):
            for path in os.scandir(self.concept["path"]):
                extension = os.path.splitext(path)[1]
                if path.is_file() \
                        and path_util.is_supported_image_extension(extension) \
                        and not path.name.endswith("-masklabel.png"):
                    preview_path = path_util.canonical_join(self.concept["path"], path.name)
                    break

        image = Image.open(preview_path)

        image_tensor = functional.to_tensor(image)
        rand = random.Random()

        if self.concept['enable_random_flip']:
            if rand.random() < 0.5:
                image_tensor = functional.hflip(image_tensor)

        if self.concept['enable_random_rotate']:
            max_angle = self.concept['random_rotate_max_angle']
            angle = rand.uniform(-max_angle, max_angle)
            image_tensor = functional.rotate(image_tensor, angle, interpolation=InterpolationMode.BILINEAR)

        if self.concept['enable_random_brightness']:
            max_strength = self.concept['random_brightness_max_strength']
            strength = rand.uniform(1 - max_strength, 1 + max_strength)
            strength = max(0.0, strength)
            image_tensor = functional.adjust_brightness(image_tensor, strength)

        if self.concept['enable_random_contrast']:
            max_strength = self.concept['random_contrast_max_strength']
            strength = rand.uniform(1 - max_strength, 1 + max_strength)
            strength = max(0.0, strength)
            image_tensor = functional.adjust_contrast(image_tensor, strength)

        if self.concept['enable_random_saturation']:
            max_strength = self.concept['random_saturation_max_strength']
            strength = rand.uniform(1 - max_strength, 1 + max_strength)
            strength = max(0.0, strength)
            image_tensor = functional.adjust_contrast(image_tensor, strength)

        if self.concept['enable_random_hue']:
            max_strength = self.concept['random_hue_max_strength']
            strength = rand.uniform(-max_strength * 0.5, max_strength * 0.5)
            strength = max(-0.5, min(0.5, strength))
            image_tensor = functional.adjust_hue(image_tensor, strength)

        image = functional.to_pil_image(image_tensor)

        size = min(image.width, image.height)
        image = image.crop((
            (image.width - size) // 2,
            (image.height - size) // 2,
            (image.width - size) // 2 + size,
            (image.height - size) // 2 + size,
        ))
        image = image.resize((300, 300), Image.Resampling.LANCZOS)

        return image

    def __ok(self):
        self.destroy()
