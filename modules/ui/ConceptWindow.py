import os
import pathlib
import random

from modules.util import path_util
from modules.util.config.ConceptConfig import ConceptConfig
from modules.util.enum.BalancingStrategy import BalancingStrategy
from modules.util.ui import components
from modules.util.ui.UIState import UIState

from mgds.LoadingPipeline import LoadingPipeline
from mgds.OutputPipelineModule import OutputPipelineModule
from mgds.PipelineModule import PipelineModule
from mgds.pipelineModules.RandomBrightness import RandomBrightness
from mgds.pipelineModules.RandomCircularMaskShrink import RandomCircularMaskShrink
from mgds.pipelineModules.RandomContrast import RandomContrast
from mgds.pipelineModules.RandomFlip import RandomFlip
from mgds.pipelineModules.RandomHue import RandomHue
from mgds.pipelineModules.RandomMaskRotateCrop import RandomMaskRotateCrop
from mgds.pipelineModules.RandomRotate import RandomRotate
from mgds.pipelineModules.RandomSaturation import RandomSaturation
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule

import torch
from torchvision.transforms import functional

import customtkinter as ctk
from PIL import Image


class InputPipelineModule(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(self, data: dict):
        super().__init__()
        self.data = data

    def length(self) -> int:
        return 1

    def get_inputs(self) -> list[str]:
        return []

    def get_outputs(self) -> list[str]:
        return list(self.data.keys())

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        return self.data


class ConceptWindow(ctk.CTkToplevel):
    def __init__(
            self,
            parent,
            concept: ConceptConfig,
            ui_state: UIState,
            image_ui_state: UIState,
            text_ui_state: UIState,
            *args, **kwargs,
    ):
        ctk.CTkToplevel.__init__(self, parent, *args, **kwargs)

        self.concept = concept
        self.ui_state = ui_state
        self.image_ui_state = image_ui_state
        self.text_ui_state = text_ui_state

        self.image_preview_file_index = 0

        self.title("Concept")
        self.geometry("800x630")
        self.resizable(True, True)
        self.wait_visibility()
        self.grab_set()
        self.focus_set()

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        tabview = ctk.CTkTabview(self)
        tabview.grid(row=0, column=0, sticky="nsew")

        self.general_tab = self.__general_tab(tabview.add("general"), concept)
        self.image_augmentation_tab = self.__image_augmentation_tab(tabview.add("image augmentation"))
        self.text_augmentation_tab = self.__text_augmentation_tab(tabview.add("text augmentation"))

        components.button(self, 1, 0, "ok", self.__ok)

    def __general_tab(self, master, concept: ConceptConfig):
        frame = ctk.CTkScrollableFrame(master, fg_color="transparent")
        frame.grid_columnconfigure(1, weight=1)
        frame.grid_columnconfigure(2, weight=1)

        # name
        components.label(frame, 0, 0, "Name",
                         tooltip="Name of the concept")
        components.entry(frame, 0, 1, self.ui_state, "name")

        # enabled
        components.label(frame, 1, 0, "Enabled",
                         tooltip="Enable or disable this concept")
        components.switch(frame, 1, 1, self.ui_state, "enabled")

        # validation_concept
        components.label(frame, 2, 0, "Validation concept",
                         tooltip="Use concept for validation instead of training")
        components.switch(frame, 2, 1, self.ui_state, "validation_concept")

        # path
        components.label(frame, 3, 0, "Path",
                         tooltip="Path where the training data is located")
        components.dir_entry(frame, 3, 1, self.ui_state, "path")

        # prompt source
        components.label(frame, 4, 0, "Prompt Source",
                         tooltip="The source for prompts used during training. When selecting \"From single text file\", select a text file that contains a list of prompts")
        prompt_path_entry = components.file_entry(frame, 4, 2, self.text_ui_state, "prompt_path")

        def set_prompt_path_entry_enabled(option: str):
            if option == 'concept':
                for child in prompt_path_entry.children.values():
                    child.configure(state="normal")
            else:
                for child in prompt_path_entry.children.values():
                    child.configure(state="disabled")

        components.options_kv(frame, 4, 1, [
            ("From text file per sample", 'sample'),
            ("From single text file", 'concept'),
            ("From image file name", 'filename'),
        ], self.text_ui_state, "prompt_source", command=set_prompt_path_entry_enabled)
        set_prompt_path_entry_enabled(concept.text.prompt_source)

        # include subdirectories
        components.label(frame, 5, 0, "Include Subdirectories",
                         tooltip="Includes images from subdirectories into the dataset")
        components.switch(frame, 5, 1, self.ui_state, "include_subdirectories")

        # image variations
        components.label(frame, 6, 0, "Image Variations",
                         tooltip="The number of different image versions to cache if latent caching is enabled.")
        components.entry(frame, 6, 1, self.ui_state, "image_variations")

        # text variations
        components.label(frame, 7, 0, "Text Variations",
                         tooltip="The number of different text versions to cache if latent caching is enabled.")
        components.entry(frame, 7, 1, self.ui_state, "text_variations")

        # balancing
        components.label(frame, 8, 0, "Balancing",
                         tooltip="The number of samples used during training. Use repeats to multiply the concept, or samples to specify an exact number of samples used in each epoch.")
        components.entry(frame, 8, 1, self.ui_state, "balancing")
        components.options(frame, 8, 2, [str(x) for x in list(BalancingStrategy)], self.ui_state, "balancing_strategy")

        # loss weight
        components.label(frame, 9, 0, "Loss Weight",
                         tooltip="The loss multiplyer for this concept.")
        components.entry(frame, 9, 1, self.ui_state, "loss_weight")

        frame.pack(fill="both", expand=1)
        return frame

    def __image_augmentation_tab(self, master):
        frame = ctk.CTkScrollableFrame(master, fg_color="transparent")
        frame.grid_columnconfigure(0, weight=0)
        frame.grid_columnconfigure(1, weight=0)
        frame.grid_columnconfigure(2, weight=0)
        frame.grid_columnconfigure(3, weight=1)

        # header
        components.label(frame, 0, 1, "Random",
                         tooltip="Enable this augmentation with random values")
        components.label(frame, 0, 2, "Fixed",
                         tooltip="Enable this augmentation with fixed values")

        # crop jitter
        components.label(frame, 1, 0, "Crop Jitter",
                         tooltip="Enables random cropping of samples")
        components.switch(frame, 1, 1, self.image_ui_state, "enable_crop_jitter")

        # random flip
        components.label(frame, 2, 0, "Random Flip",
                         tooltip="Randomly flip the sample during training")
        components.switch(frame, 2, 1, self.image_ui_state, "enable_random_flip")
        components.switch(frame, 2, 2, self.image_ui_state, "enable_fixed_flip")

        # random rotation
        components.label(frame, 3, 0, "Random Rotation",
                         tooltip="Randomly rotates the sample during training")
        components.switch(frame, 3, 1, self.image_ui_state, "enable_random_rotate")
        components.switch(frame, 3, 2, self.image_ui_state, "enable_fixed_rotate")
        components.entry(frame, 3, 3, self.image_ui_state, "random_rotate_max_angle")

        # random brightness
        components.label(frame, 4, 0, "Random Brightness",
                         tooltip="Randomly adjusts the brightness of the sample during training")
        components.switch(frame, 4, 1, self.image_ui_state, "enable_random_brightness")
        components.switch(frame, 4, 2, self.image_ui_state, "enable_fixed_brightness")
        components.entry(frame, 4, 3, self.image_ui_state, "random_brightness_max_strength")

        # random contrast
        components.label(frame, 5, 0, "Random Contrast",
                         tooltip="Randomly adjusts the contrast of the sample during training")
        components.switch(frame, 5, 1, self.image_ui_state, "enable_random_contrast")
        components.switch(frame, 5, 2, self.image_ui_state, "enable_fixed_contrast")
        components.entry(frame, 5, 3, self.image_ui_state, "random_contrast_max_strength")

        # random saturation
        components.label(frame, 6, 0, "Random Saturation",
                         tooltip="Randomly adjusts the saturation of the sample during training")
        components.switch(frame, 6, 1, self.image_ui_state, "enable_random_saturation")
        components.switch(frame, 6, 2, self.image_ui_state, "enable_fixed_saturation")
        components.entry(frame, 6, 3, self.image_ui_state, "random_saturation_max_strength")

        # random hue
        components.label(frame, 7, 0, "Random Hue",
                         tooltip="Randomly adjusts the hue of the sample during training")
        components.switch(frame, 7, 1, self.image_ui_state, "enable_random_hue")
        components.switch(frame, 7, 2, self.image_ui_state, "enable_fixed_hue")
        components.entry(frame, 7, 3, self.image_ui_state, "random_hue_max_strength")

        # random circular mask shrink
        components.label(frame, 8, 0, "Circular Mask Generation",
                         tooltip="Automatically create circular masks for masked training")
        components.switch(frame, 8, 1, self.image_ui_state, "enable_random_circular_mask_shrink")

        # random rotate and crop
        components.label(frame, 9, 0, "Random Rotate and Crop",
                         tooltip="Randomly rotate the training samples and crop to the masked region")
        components.switch(frame, 9, 1, self.image_ui_state, "enable_random_mask_rotate_crop")

        # circular mask generation
        components.label(frame, 10, 0, "Resolution Override",
                         tooltip="Override the resolution for this concept. Optionally specify multiple resolutions separated by a comma, or a single exact resolution in the format <width>x<height>")
        components.switch(frame, 10, 2, self.image_ui_state, "enable_resolution_override")
        components.entry(frame, 10, 3, self.image_ui_state, "resolution_override")

        # image
        image_preview, filename_preview, caption_preview = self.__get_preview_image()
        self.image = ctk.CTkImage(
            light_image=image_preview,
            size=image_preview.size,
        )
        image_label = ctk.CTkLabel(master=frame, text="", image=self.image, height=300, width=300)
        image_label.grid(row=0, column=4, rowspan=6)

        # refresh preview
        update_button_frame = ctk.CTkFrame(master=frame, corner_radius=0, fg_color="transparent")
        update_button_frame.grid(row=6, column=4, sticky="nsew")
        update_button_frame.grid_columnconfigure(1, weight=1)

        prev_preview_button = components.button(update_button_frame, 0, 0, "<", command=self.__prev_image_preview)
        components.button(update_button_frame, 0, 1, "Update Preview", command=self.__update_image_preview)
        next_preview_button = components.button(update_button_frame, 0, 2, ">", command=self.__next_image_preview)

        prev_preview_button.configure(width=40)
        next_preview_button.configure(width=40)

        #caption and filename preview
        self.filename_preview = ctk.CTkLabel(master=frame, text=filename_preview, width=300, anchor="nw", justify="left", padx=10, wraplength=280)
        self.filename_preview.grid(row=7, column=4)
        self.caption_preview = ctk.CTkTextbox(master=frame, width = 300, height = 150, wrap="word",
                                              bg_color="white", border_width=3, corner_radius=3, border_color="#3B8ED0")
        self.caption_preview.insert(index="1.0", text=caption_preview)
        self.caption_preview.configure(state="disabled")
        self.caption_preview.grid(row=8, column=4, rowspan = 4)

        frame.pack(fill="both", expand=1)
        return frame

    def __text_augmentation_tab(self, master):
        frame = ctk.CTkScrollableFrame(master, fg_color="transparent")
        frame.grid_columnconfigure(0, weight=0)
        frame.grid_columnconfigure(1, weight=0)
        frame.grid_columnconfigure(2, weight=0)
        frame.grid_columnconfigure(3, weight=1)

        # tag shuffling
        components.label(frame, 0, 0, "Tag Shuffling",
                         tooltip="Enables tag shuffling")
        components.switch(frame, 0, 1, self.text_ui_state, "enable_tag_shuffling")

        # keep tag count
        components.label(frame, 1, 0, "Tag Delimiter",
                         tooltip="The delimiter between tags")
        components.entry(frame, 1, 1, self.text_ui_state, "tag_delimiter")

        # keep tag count
        components.label(frame, 2, 0, "Keep Tag Count",
                         tooltip="The number of tags at the start of the caption that are not shuffled or dropped")
        components.entry(frame, 2, 1, self.text_ui_state, "keep_tags_count")

        # tag dropout
        components.label(frame, 3, 0, "Tag Dropout",
                         tooltip="Enables random dropout for tags in the captions.")
        components.switch(frame, 3, 1, self.text_ui_state, "tag_dropout_enable")
        components.label(frame, 4, 0, "Dropout Mode",
                         tooltip="Method used to drop captions. 'Full' will drop the entire caption past the 'kept' tags with a certain probability, 'Random' will drop individual tags with the set probability, and 'Random Weighted' will linearly increase the probability of dropping tags, more likely to preseve tags near the front with full probability to drop at the end.")
        components.options_kv(frame, 4, 1, [
            ("Full", 'FULL'),
            ("Random", 'RANDOM'),
            ("Random Weighted", 'RANDOM WEIGHTED'),
        ], self.text_ui_state, "tag_dropout_mode", None)
        components.label(frame, 4, 2, "Probability",
                         tooltip="Probability to drop tags, from 0 to 1.")
        components.entry(frame, 4, 3, self.text_ui_state, "tag_dropout_probability")

        components.label(frame, 5, 0, "Special Dropout Tags",
                         tooltip="List of tags which will be whitelisted/blacklisted by dropout. 'Whitelist' tags will never be dropped but all others may be, 'Blacklist' tags may be dropped but all others will never be, 'None' may drop any tags. Can specify either a delimiter-separated list in the field, or a file path to a .txt or .csv file with entries separated by newlines.")
        components.options_kv(frame, 5, 1, [
            ("None", 'NONE'),
            ("Blacklist", 'BLACKLIST'),
            ("Whitelist", 'WHITELIST'),
        ], self.text_ui_state, "tag_dropout_special_tags_mode", None)
        components.entry(frame, 5, 2, self.text_ui_state, "tag_dropout_special_tags")
        components.label(frame, 6, 0, "Special Tags Regex",
                         tooltip="Interpret special tags with regex, such as 'photo.*' to match 'photo, photograph, photon' but not 'telephoto'. Includes exception for '/(' and '/)' syntax found in many booru/e6 tags.")
        components.switch(frame, 6, 1, self.text_ui_state, "tag_dropout_special_tags_regex")

        #capitalization randomization
        components.label(frame, 7, 0, "Randomize Capitalization",
                         tooltip="Enables randomization of capitalization for tags in the caption.")
        components.switch(frame, 7, 1, self.text_ui_state, "caps_randomize_enable")
        components.label(frame, 7, 2, "Force Lowercase",
                         tooltip="If enabled, converts the caption to lowercase before any further processing.")
        components.switch(frame, 7, 3, self.text_ui_state, "caps_randomize_lowercase")

        components.label(frame, 8, 0, "Captialization Mode",
                         tooltip="Comma-separated list of types of capitalization randomization to perform. 'capslock' for ALL CAPS, 'title' for First Letter Of Every Word, 'first' for First word only, 'random' for rAndOMiZeD lEtTERs.")
        components.entry(frame, 8, 1, self.text_ui_state, "caps_randomize_mode")
        components.label(frame, 8, 2, "Probability",
                         tooltip="Probability to randomize capitialization of each tag, from 0 to 1.")
        components.entry(frame, 8, 3, self.text_ui_state, "caps_randomize_probability")

        frame.pack(fill="both", expand=1)
        return frame

    def __prev_image_preview(self):
        self.image_preview_file_index = max(self.image_preview_file_index - 1, 0)
        self.__update_image_preview()

    def __next_image_preview(self):
        self.image_preview_file_index += 1
        self.__update_image_preview()

    def __update_image_preview(self):
        image_preview, filename_preview, caption_preview = self.__get_preview_image()
        self.image.configure(light_image=image_preview, size=image_preview.size)
        self.filename_preview.configure(text=filename_preview)
        self.caption_preview.configure(state="normal")
        self.caption_preview.delete(index1="1.0", index2="end")
        self.caption_preview.insert(index="1.0", text=caption_preview)
        self.caption_preview.configure(state="disabled")

    def __get_preview_image(self):
        preview_image_path = "resources/icons/icon.png"
        file_index = -1
        glob_pattern = "**/*.*" if self.concept.include_subdirectories else "*.*"

        if os.path.isdir(self.concept.path):
            for path in pathlib.Path(self.concept.path).glob(glob_pattern):
                extension = os.path.splitext(path)[1]
                if path.is_file() and path_util.is_supported_image_extension(extension) \
                        and not path.name.endswith("-masklabel.png"):
                    preview_image_path = path_util.canonical_join(self.concept.path, path)

                    file_index += 1
                    if file_index == self.image_preview_file_index:
                        break


        image = Image.open(preview_image_path).convert("RGB")
        image_tensor = functional.to_tensor(image)

        splitext = os.path.splitext(preview_image_path)
        preview_mask_path = path_util.canonical_join(splitext[0] + "-masklabel.png")
        if not os.path.isfile(preview_mask_path):
            preview_mask_path = None

        if preview_mask_path:
            mask = Image.open(preview_mask_path).convert("L")
            mask_tensor = functional.to_tensor(mask)
        else:
            mask_tensor = torch.ones((1, image_tensor.shape[1], image_tensor.shape[2]))

        input_module = InputPipelineModule({
            'true': True,
            'image': image_tensor,
            'mask': mask_tensor,
            'enable_random_flip': self.concept.image.enable_random_flip,
            'enable_fixed_flip': self.concept.image.enable_fixed_flip,
            'enable_random_rotate': self.concept.image.enable_random_rotate,
            'enable_fixed_rotate': self.concept.image.enable_fixed_rotate,
            'random_rotate_max_angle': self.concept.image.random_rotate_max_angle,
            'enable_random_brightness': self.concept.image.enable_random_brightness,
            'enable_fixed_brightness': self.concept.image.enable_fixed_brightness,
            'random_brightness_max_strength': self.concept.image.random_brightness_max_strength,
            'enable_random_contrast': self.concept.image.enable_random_contrast,
            'enable_fixed_contrast': self.concept.image.enable_fixed_contrast,
            'random_contrast_max_strength': self.concept.image.random_contrast_max_strength,
            'enable_random_saturation': self.concept.image.enable_random_saturation,
            'enable_fixed_saturation': self.concept.image.enable_fixed_saturation,
            'random_saturation_max_strength': self.concept.image.random_saturation_max_strength,
            'enable_random_hue': self.concept.image.enable_random_hue,
            'enable_fixed_hue': self.concept.image.enable_fixed_hue,
            'random_hue_max_strength': self.concept.image.random_hue_max_strength,
            'enable_random_circular_mask_shrink': self.concept.image.enable_random_circular_mask_shrink,
            'enable_random_mask_rotate_crop': self.concept.image.enable_random_mask_rotate_crop,
        })

        circular_mask_shrink = RandomCircularMaskShrink(mask_name='mask', shrink_probability=1.0, shrink_factor_min=0.2, shrink_factor_max=1.0, enabled_in_name='enable_random_circular_mask_shrink')
        random_mask_rotate_crop = RandomMaskRotateCrop(mask_name='mask', additional_names=['image'], min_size=512, min_padding_percent=10, max_padding_percent=30, max_rotate_angle=20, enabled_in_name='enable_random_mask_rotate_crop')
        random_flip = RandomFlip(names=['image', 'mask'], enabled_in_name='enable_random_flip', fixed_enabled_in_name='enable_fixed_flip')
        random_rotate = RandomRotate(names=['image', 'mask'], enabled_in_name='enable_random_rotate', fixed_enabled_in_name='enable_fixed_rotate', max_angle_in_name='random_rotate_max_angle')
        random_brightness = RandomBrightness(names=['image'], enabled_in_name='enable_random_brightness', fixed_enabled_in_name='enable_fixed_brightness', max_strength_in_name='random_brightness_max_strength')
        random_contrast = RandomContrast(names=['image'], enabled_in_name='enable_random_contrast', fixed_enabled_in_name='enable_fixed_contrast', max_strength_in_name='random_contrast_max_strength')
        random_saturation = RandomSaturation(names=['image'], enabled_in_name='enable_random_saturation', fixed_enabled_in_name='enable_fixed_saturation', max_strength_in_name='random_saturation_max_strength')
        random_hue = RandomHue(names=['image'], enabled_in_name='enable_random_hue', fixed_enabled_in_name='enable_fixed_hue', max_strength_in_name='random_hue_max_strength')
        output_module = OutputPipelineModule(['image', 'mask'])

        modules = [
            input_module,
            circular_mask_shrink,
            random_mask_rotate_crop,
            random_flip,
            random_rotate,
            random_brightness,
            random_contrast,
            random_saturation,
            random_hue,
            output_module,
        ]

        pipeline = LoadingPipeline(
            device=torch.device('cpu'),
            modules=modules,
            batch_size=1,
            seed=random.randint(0, 2**30),
            state=None,
            initial_epoch=0,
            initial_index=0,
        )

        data = pipeline.__next__()
        image_tensor = data['image']
        mask_tensor = data['mask']
        #display filename and first line of base caption from prompt source
        #will try to change to preview caption with text variations at some point
        filename_output = os.path.basename(preview_image_path)
        try:
            if self.concept.text.prompt_source == "sample":
                with open(splitext[0] + ".txt") as prompt_file:
                    prompt_output = prompt_file.readline()
            elif self.concept.text.prompt_source == "filename":
                prompt_output = os.path.splitext(os.path.basename(preview_image_path))[0]
            elif self.concept.text.prompt_source == "concept":
                with open(self.concept.text.prompt_path) as prompt_file:
                    prompt_output = prompt_file.readline()
        except FileNotFoundError:
            prompt_output = "No caption found."

        mask_tensor = torch.clamp(mask_tensor, 0.3, 1)
        image_tensor = image_tensor * mask_tensor

        image = functional.to_pil_image(image_tensor)

        image.thumbnail((300, 300))

        return image, filename_output, prompt_output

    def __ok(self):
        self.destroy()
