import fractions
import math
import os
import pathlib
import platform
import random
import threading
import time
import traceback

from modules.util import concept_stats, path_util
from modules.util.config.ConceptConfig import ConceptConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.BalancingStrategy import BalancingStrategy
from modules.util.enum.ConceptType import ConceptType
from modules.util.image_util import load_image
from modules.util.ui import components
from modules.util.ui.ui_utils import set_window_icon
from modules.util.ui.UIState import UIState

from mgds.LoadingPipeline import LoadingPipeline
from mgds.OutputPipelineModule import OutputPipelineModule
from mgds.PipelineModule import PipelineModule
from mgds.pipelineModules.CapitalizeTags import CapitalizeTags
from mgds.pipelineModules.DropTags import DropTags
from mgds.pipelineModules.RandomBrightness import RandomBrightness
from mgds.pipelineModules.RandomCircularMaskShrink import (
    RandomCircularMaskShrink,
)
from mgds.pipelineModules.RandomContrast import RandomContrast
from mgds.pipelineModules.RandomFlip import RandomFlip
from mgds.pipelineModules.RandomHue import RandomHue
from mgds.pipelineModules.RandomMaskRotateCrop import RandomMaskRotateCrop
from mgds.pipelineModules.RandomRotate import RandomRotate
from mgds.pipelineModules.RandomSaturation import RandomSaturation
from mgds.pipelineModules.ShuffleTags import ShuffleTags
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import (
    RandomAccessPipelineModule,
)

import torch
from torchvision.transforms import functional

import customtkinter as ctk
import huggingface_hub
from customtkinter import AppearanceModeTracker, ThemeManager
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
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
            train_config: TrainConfig,
            concept: ConceptConfig,
            ui_state: UIState,
            image_ui_state: UIState,
            text_ui_state: UIState,
            *args, **kwargs,
    ):
        super().__init__(parent, *args, **kwargs)

        self.train_config = train_config

        self.concept = concept
        self.ui_state = ui_state
        self.image_ui_state = image_ui_state
        self.text_ui_state = text_ui_state
        self.image_preview_file_index = 0
        self.preview_augmentations = ctk.BooleanVar(self, True)

        self.title("Concept")
        self.geometry("800x700")
        self.resizable(True, True)

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        tabview = ctk.CTkTabview(self)
        tabview.grid(row=0, column=0, sticky="nsew")

        self.general_tab = self.__general_tab(tabview.add("general"), concept)
        self.image_augmentation_tab = self.__image_augmentation_tab(tabview.add("image augmentation"))
        self.text_augmentation_tab = self.__text_augmentation_tab(tabview.add("text augmentation"))
        self.concept_stats_tab = self.__concept_stats_tab(tabview.add("statistics"))

        #automatic concept scan
        self.scan_thread = threading.Thread(target=self.__auto_update_concept_stats, daemon=True)
        self.scan_thread.start()

        components.button(self, 1, 0, "ok", self.__ok)

        self.wait_visibility()
        self.grab_set()
        self.focus_set()
        self.after(200, lambda: set_window_icon(self))


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

        # concept type
        components.label(frame, 2, 0, "Concept Type",
                         tooltip="STANDARD: Standard finetuning with the sample as training target\n"
                                 "VALIDATION: Use concept for validation instead of training\n"
                                 "PRIOR_PREDICTION: Use the sample to make a prediction using the model as it was before training. This prediction is then used as the training target "
                                 "for the model in training. This can be used as regularisation and to preserve prior model knowledge while finetuning the model on other concepts. "
                                 "Only implemented for LoRA.",
                         wide_tooltip=True)
        components.options(frame, 2, 1, [str(x) for x in list(ConceptType)], self.ui_state, "type")

        # path
        components.label(frame, 3, 0, "Path",
                         tooltip="Path where the training data is located")
        components.dir_entry(frame, 3, 1, self.ui_state, "path")
        components.button(frame, 3, 2, text="download now", command=self.__download_dataset_threaded,
                          tooltip="Download dataset from Huggingface now, for the purpose of previewing and statistics. Otherwise, it will be downloaded when you start training. Path must be a Huggingface repository.")

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
        update_button_frame.grid(row=6, column=4, rowspan=6, sticky="nsew")
        update_button_frame.grid_columnconfigure(1, weight=1)

        prev_preview_button = components.button(update_button_frame, 0, 0, "<", command=self.__prev_image_preview)
        components.button(update_button_frame, 0, 1, "Update Preview", command=self.__update_image_preview)
        next_preview_button = components.button(update_button_frame, 0, 2, ">", command=self.__next_image_preview)
        preview_augmentations_switch = ctk.CTkSwitch(update_button_frame, text="Show Augmentations", variable=self.preview_augmentations, command=self.__update_image_preview)
        preview_augmentations_switch.grid(row=1, column=0, columnspan=3, padx=5, pady=5)

        prev_preview_button.configure(width=40)
        next_preview_button.configure(width=40)

        #caption and filename preview
        self.filename_preview = ctk.CTkLabel(master=update_button_frame, text=filename_preview, width=300, anchor="nw", justify="left", padx=10, wraplength=280)
        self.filename_preview.grid(row=2, column=0, columnspan=3)
        self.caption_preview = ctk.CTkTextbox(master=update_button_frame, width = 300, height = 150, wrap="word", border_width=2)
        self.caption_preview.insert(index="1.0", text=caption_preview)
        self.caption_preview.configure(state="disabled")
        self.caption_preview.grid(row=3, column=0, columnspan=3, rowspan=3)

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

    def __concept_stats_tab(self, master):
        frame = ctk.CTkScrollableFrame(master, fg_color="transparent")
        frame.grid_columnconfigure(0, weight=0, minsize=150)
        frame.grid_columnconfigure(1, weight=0, minsize=150)
        frame.grid_columnconfigure(2, weight=0, minsize=150)
        frame.grid_columnconfigure(3, weight=0, minsize=150)

        self.cancel_scan_flag = threading.Event()

        #file size
        self.file_size_label = components.label(frame, 1, 0, "Total Size", pad=0,
                         tooltip="Total size of all image, mask, and caption files in MB")
        self.file_size_label.configure(font=ctk.CTkFont(underline=True))
        self.file_size_preview = components.label(frame, 2, 0, pad=0, text="-")

        #subdirectory count
        self.dir_count_label = components.label(frame, 1, 1, "Directories", pad=0,
                         tooltip="Total number of directories including and under (if 'include subdirectories' is enabled) the main concept directory")
        self.dir_count_label.configure(font=ctk.CTkFont(underline=True))
        self.dir_count_preview = components.label(frame, 2, 1, pad=0, text="-")

        #basic img/vid stats - count of each type in the concept
        #the \n at the start of the label gives it better vertical spacing with other rows
        self.image_count_label = components.label(frame, 3, 0, "\nTotal Images", pad=0,
                         tooltip="Total number of image files, any of the extensions " + str(path_util.SUPPORTED_IMAGE_EXTENSIONS) + ", excluding '-masklabel.png and -condlabel.png'")
        self.image_count_label.configure(font=ctk.CTkFont(underline=True))
        self.image_count_preview = components.label(frame, 4, 0, pad=0, text="-")
        self.video_count_label = components.label(frame, 3, 1, "\nTotal Videos", pad=0,
                         tooltip="Total number of video files, any of the extensions " + str(path_util.SUPPORTED_VIDEO_EXTENSIONS))
        self.video_count_label.configure(font=ctk.CTkFont(underline=True))
        self.video_count_preview = components.label(frame, 4, 1, pad=0, text="-")
        self.mask_count_label = components.label(frame, 3, 2, "\nTotal Masks", pad=0,
                         tooltip="Total number of mask files, any file ending in '-masklabel.png'")
        self.mask_count_label.configure(font=ctk.CTkFont(underline=True))
        self.mask_count_preview = components.label(frame, 4, 2, pad=0, text="-")
        self.caption_count_label = components.label(frame, 3, 3, "\nTotal Captions", pad=0,
                         tooltip="Total number of caption files, any .txt file. With advanced scan, includes the total number of captions on separate lines across all files in parentheses.")
        self.caption_count_label.configure(font=ctk.CTkFont(underline=True))
        self.caption_count_preview = components.label(frame, 4, 3, pad=0, text="-")

        #advanced img/vid stats - how many img/vid files have a mask or caption of the same name
        self.image_count_mask_label = components.label(frame, 5, 0, "\nImages with Masks", pad=0,
                         tooltip="Total number of image files with an associated mask")
        self.image_count_mask_label.configure(font=ctk.CTkFont(underline=True))
        self.image_count_mask_preview = components.label(frame, 6, 0, pad=0, text="-")
        self.mask_count_label_unpaired = components.label(frame, 5, 1, "\nUnpaired Masks", pad=0,
                         tooltip="Total number of mask files which lack a corresponding image file - if >0, check your data set!")
        self.mask_count_label_unpaired.configure(font=ctk.CTkFont(underline=True))
        self.mask_count_preview_unpaired = components.label(frame, 6, 1, pad=0, text="-")
        #currently no masks for videos?

        self.image_count_caption_label = components.label(frame, 7, 0, "\nImages with Captions", pad=0,
                         tooltip="Total number of image files with an associated caption")
        self.image_count_caption_label.configure(font=ctk.CTkFont(underline=True))
        self.image_count_caption_preview = components.label(frame, 8, 0, pad=0, text="-")
        self.video_count_caption_label = components.label(frame, 7, 1, "\nVideos with Captions", pad=0,
                         tooltip="Total number of video files with an associated caption")
        self.video_count_caption_label.configure(font=ctk.CTkFont(underline=True))
        self.video_count_caption_preview = components.label(frame, 8, 1, pad=0, text="-")
        self.caption_count_label_unpaired = components.label(frame, 7, 2, "\nUnpaired Captions", pad=0,
                         tooltip="Total number of caption files which lack a corresponding image file - if >0, check your data set! If using 'from file name' or 'from single text file' then this can be ignored.")
        self.caption_count_label_unpaired.configure(font=ctk.CTkFont(underline=True))
        self.caption_count_preview_unpaired = components.label(frame, 8, 2, pad=0, text="-")

        #resolution info
        self.pixel_max_label = components.label(frame, 9, 0, "\nMax Pixels", pad=0,
                         tooltip="Largest image in the concept by number of pixels (width * height)")
        self.pixel_max_label.configure(font=ctk.CTkFont(underline=True))
        self.pixel_max_preview = components.label(frame, 10, 0, pad=0, text="-", wraplength=150)
        self.pixel_avg_label = components.label(frame, 9, 1, "\nAvg Pixels", pad=0,
                         tooltip="Average size of images in the concept by number of pixels (width * height)")
        self.pixel_avg_label.configure(font=ctk.CTkFont(underline=True))
        self.pixel_avg_preview = components.label(frame, 10, 1, pad=0, text="-", wraplength=150)
        self.pixel_min_label = components.label(frame, 9, 2, "\nMin Pixels", pad=0,
                         tooltip="Smallest image in the concept by number of pixels (width * height)")
        self.pixel_min_label.configure(font=ctk.CTkFont(underline=True))
        self.pixel_min_preview = components.label(frame, 10, 2, pad=0, text="-", wraplength=150)

        #video length info
        self.length_max_label = components.label(frame, 11, 0, "\nMax Length", pad=0,
                         tooltip="Longest video in the concept by number of frames")
        self.length_max_label.configure(font=ctk.CTkFont(underline=True))
        self.length_max_preview = components.label(frame, 12, 0, pad=0, text="-", wraplength=150)
        self.length_avg_label = components.label(frame, 11, 1, "\nAvg Length", pad=0,
                         tooltip="Average length of videos in the concept by number of frames")
        self.length_avg_label.configure(font=ctk.CTkFont(underline=True))
        self.length_avg_preview = components.label(frame, 12, 1, pad=0, text="-", wraplength=150)
        self.length_min_label = components.label(frame, 11, 2, "\nMin Length", pad=0,
                         tooltip="Shortest video in the concept by number of frames")
        self.length_min_label.configure(font=ctk.CTkFont(underline=True))
        self.length_min_preview = components.label(frame, 12, 2, pad=0, text="-", wraplength=150)

        #video fps info
        self.fps_max_label = components.label(frame, 13, 0, "\nMax FPS", pad=0,
                         tooltip="Video in concept with highest fps")
        self.fps_max_label.configure(font=ctk.CTkFont(underline=True))
        self.fps_max_preview = components.label(frame, 14, 0, pad=0, text="-", wraplength=150)
        self.fps_avg_label = components.label(frame, 13, 1, "\nAvg FPS", pad=0,
                         tooltip="Average fps of videos in the concept")
        self.fps_avg_label.configure(font=ctk.CTkFont(underline=True))
        self.fps_avg_preview = components.label(frame, 14, 1, pad=0, text="-", wraplength=150)
        self.fps_min_label = components.label(frame, 13, 2, "\nMin FPS", pad=0,
                         tooltip="Video in concept with the lowest fps")
        self.fps_min_label.configure(font=ctk.CTkFont(underline=True))
        self.fps_min_preview = components.label(frame, 14, 2, pad=0, text="-", wraplength=150)

        #caption info
        self.caption_max_label = components.label(frame, 15, 0, "\nMax Caption Length", pad=0,
                         tooltip="Largest caption in concept by character count. For token count, assume ~2 tokens/word")
        self.caption_max_label.configure(font=ctk.CTkFont(underline=True))
        self.caption_max_preview = components.label(frame, 16, 0, pad=0, text="-", wraplength=150)
        self.caption_avg_label = components.label(frame, 15, 1, "\nAvg Caption Length", pad=0,
                         tooltip="Average length of caption in concept by character count. For token count, assume ~2 tokens/word")
        self.caption_avg_label.configure(font=ctk.CTkFont(underline=True))
        self.caption_avg_preview = components.label(frame, 16, 1, pad=0, text="-", wraplength=150)
        self.caption_min_label = components.label(frame, 15, 2, "\nMin Caption Length", pad=0,
                         tooltip="Smallest caption in concept by character count. For token count, assume ~2 tokens/word")
        self.caption_min_label.configure(font=ctk.CTkFont(underline=True))
        self.caption_min_preview = components.label(frame, 16, 2, pad=0, text="-", wraplength=150)

        #aspect bucket info
        self.aspect_bucket_label = components.label(frame, 17, 0, "\nAspect Bucketing", pad=0,
                         tooltip="Graph of all possible buckets and the number of images in each one, defined as height/width. Buckets range from 0.25 (4:1 extremely wide) to 4 (1:4 extremely tall). \
                            Images which don't match a bucket exactly are cropped to the nearest one.")
        self.aspect_bucket_label.configure(font=ctk.CTkFont(underline=True))
        self.small_bucket_label = components.label(frame, 17, 1, "\nSmallest Buckets", pad=0,
                         tooltip="Image buckets with the least nonzero total images - if 'batch size' is larger than this, these images will be ignored during training! See the wiki for more details.")
        self.small_bucket_label.configure(font=ctk.CTkFont(underline=True))
        self.small_bucket_preview = components.label(frame, 18, 1, pad=0, text="-")

        #aspect bucketing plot, mostly copied from timestep preview graph
        appearance_mode = AppearanceModeTracker.get_mode()
        background_color = self.winfo_rgb(ThemeManager.theme["CTkToplevel"]["fg_color"][appearance_mode])
        text_color = self.winfo_rgb(ThemeManager.theme["CTkLabel"]["text_color"][appearance_mode])
        background_color = f"#{int(background_color[0]/256):x}{int(background_color[1]/256):x}{int(background_color[2]/256):x}"
        self.text_color = f"#{int(text_color[0]/256):x}{int(text_color[1]/256):x}{int(text_color[2]/256):x}"

        plt.set_loglevel('WARNING')     #suppress errors about data type in bar chart
        self.bucket_fig, self.bucket_ax = plt.subplots(figsize=(7,3))
        self.canvas = FigureCanvasTkAgg(self.bucket_fig, master=frame)
        self.canvas.get_tk_widget().grid(row=19, column=0, columnspan=4, rowspan=2)
        self.bucket_fig.tight_layout()
        self.bucket_fig.subplots_adjust(bottom=0.15)

        self.bucket_fig.set_facecolor(background_color)
        self.bucket_ax.set_facecolor(background_color)
        self.bucket_ax.spines['bottom'].set_color(self.text_color)
        self.bucket_ax.spines['left'].set_color(self.text_color)
        self.bucket_ax.spines['top'].set_visible(False)
        self.bucket_ax.spines['right'].set_color(self.text_color)
        self.bucket_ax.tick_params(axis='x', colors=self.text_color, which="both")
        self.bucket_ax.tick_params(axis='y', colors=self.text_color, which="both")
        self.bucket_ax.xaxis.label.set_color(self.text_color)
        self.bucket_ax.yaxis.label.set_color(self.text_color)

        #refresh stats - must be after all labels are defined or will give error
        self.refresh_basic_stats_button = components.button(master=frame, row=0, column=0, text="Refresh Basic", command=lambda: self.__get_concept_stats_threaded(False, 9999),
                          tooltip="Reload basic statistics for the concept directory")
        self.refresh_advanced_stats_button = components.button(master=frame, row=0, column=1, text="Refresh Advanced", command=lambda: self.__get_concept_stats_threaded(True, 9999),
                          tooltip="Reload advanced statistics for the concept directory")       #run "basic" scan first before "advanced", seems to help the system cache the directories and run faster
        self.cancel_stats_button = components.button(master=frame, row=0, column=2, text="Abort Scan", command=lambda: self.__cancel_concept_stats(),
                          tooltip="Stop the currently running scan if it's taking a long time - advanced scan will be slow on large folders and on HDDs")
        self.processing_time = components.label(frame, 0, 3, text="-", tooltip="Time taken to process concept directory")

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

    @staticmethod
    def get_concept_path(path: str) -> str | None:
        if os.path.isdir(path):
            return path
        try:
            #don't download, only check if available locally:
            return huggingface_hub.snapshot_download(repo_id=path, repo_type="dataset", local_files_only=True)
        except Exception:
            return None

    def __download_dataset(self):
        try:
            huggingface_hub.login(token=self.train_config.secrets.huggingface_token, new_session=False)
            huggingface_hub.snapshot_download(repo_id=self.concept.path, repo_type="dataset")
        except Exception:
            traceback.print_exc()

    def __download_dataset_threaded(self):
        download_thread = threading.Thread(target=self.__download_dataset, daemon=True)
        download_thread.start()

    def _read_text_file_for_preview(self, file_path: str) -> str:
        empty_msg = "[Empty prompt]"
        try:
            with open(file_path, "r") as f:
                if self.preview_augmentations.get():
                    lines = [line.strip() for line in f if line.strip()]
                    return random.choice(lines) if lines else empty_msg
                content = f.read().strip()
                return content if content else empty_msg
        except FileNotFoundError:
            return "File not found, please check the path"
        except IsADirectoryError:
            return "[Provided path is a directory, please correct the caption path]"
        except PermissionError:
            if platform.system() == "Windows":
                return "[Permission denied, please check the file permissions or Windows Defender settings]"
            else:
                return "[Permission denied, please check the file permissions]"
        except UnicodeDecodeError:
            return "[Invalid file encoding. This should not happen, please report this issue]"

    def __get_preview_image(self):
        preview_image_path = "resources/icons/icon.png"
        file_index = -1
        glob_pattern = "**/*.*" if self.concept.include_subdirectories else "*.*"

        concept_path = self.get_concept_path(self.concept.path)
        if concept_path:
            for path in pathlib.Path(concept_path).glob(glob_pattern):
                extension = os.path.splitext(path)[1]
                if path.is_file() and path_util.is_supported_image_extension(extension) \
                        and not path.name.endswith("-masklabel.png") and not path.name.endswith("-condlabel.png"):
                    preview_image_path = path_util.canonical_join(concept_path, path)
                    file_index += 1
                    if file_index == self.image_preview_file_index:
                        break

        image = load_image(preview_image_path, 'RGB')
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

        source = self.concept.text.prompt_source
        preview_p = pathlib.Path(preview_image_path)
        if source == "filename":
            prompt_output = preview_p.stem or "[Empty prompt]"
        else:
            file_map = {
                "sample": preview_p.with_suffix(".txt"),
                "concept": pathlib.Path(self.concept.text.prompt_path) if self.concept.text.prompt_path else None,
            }
            file_path = file_map.get(source)
            prompt_output = self._read_text_file_for_preview(str(file_path)) if file_path else "[Empty prompt]"

        modules = []
        if self.preview_augmentations.get():
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

                'prompt' : prompt_output,
                'tag_dropout_enable' : self.concept.text.tag_dropout_enable,
                'tag_dropout_probability' : self.concept.text.tag_dropout_probability,
                'tag_dropout_mode' : self.concept.text.tag_dropout_mode,
                'tag_dropout_special_tags' : self.concept.text.tag_dropout_special_tags,
                'tag_dropout_special_tags_mode' : self.concept.text.tag_dropout_special_tags_mode,
                'tag_delimiter' : self.concept.text.tag_delimiter,
                'keep_tags_count' : self.concept.text.keep_tags_count,
                'tag_dropout_special_tags_regex' : self.concept.text.tag_dropout_special_tags_regex,
                'caps_randomize_enable' : self.concept.text.caps_randomize_enable,
                'caps_randomize_probability' : self.concept.text.caps_randomize_probability,
                'caps_randomize_mode' : self.concept.text.caps_randomize_mode,
                'caps_randomize_lowercase' : self.concept.text.caps_randomize_lowercase,
                'enable_tag_shuffling' : self.concept.text.enable_tag_shuffling,
            })

            circular_mask_shrink = RandomCircularMaskShrink(mask_name='mask', shrink_probability=1.0, shrink_factor_min=0.2, shrink_factor_max=1.0, enabled_in_name='enable_random_circular_mask_shrink')
            random_mask_rotate_crop = RandomMaskRotateCrop(mask_name='mask', additional_names=['image'], min_size=512, min_padding_percent=10, max_padding_percent=30, max_rotate_angle=20, enabled_in_name='enable_random_mask_rotate_crop')
            random_flip = RandomFlip(names=['image', 'mask'], enabled_in_name='enable_random_flip', fixed_enabled_in_name='enable_fixed_flip')
            random_rotate = RandomRotate(names=['image', 'mask'], enabled_in_name='enable_random_rotate', fixed_enabled_in_name='enable_fixed_rotate', max_angle_in_name='random_rotate_max_angle')
            random_brightness = RandomBrightness(names=['image'], enabled_in_name='enable_random_brightness', fixed_enabled_in_name='enable_fixed_brightness', max_strength_in_name='random_brightness_max_strength')
            random_contrast = RandomContrast(names=['image'], enabled_in_name='enable_random_contrast', fixed_enabled_in_name='enable_fixed_contrast', max_strength_in_name='random_contrast_max_strength')
            random_saturation = RandomSaturation(names=['image'], enabled_in_name='enable_random_saturation', fixed_enabled_in_name='enable_fixed_saturation', max_strength_in_name='random_saturation_max_strength')
            random_hue = RandomHue(names=['image'], enabled_in_name='enable_random_hue', fixed_enabled_in_name='enable_fixed_hue', max_strength_in_name='random_hue_max_strength')
            drop_tags = DropTags(text_in_name='prompt', enabled_in_name='tag_dropout_enable', probability_in_name='tag_dropout_probability', dropout_mode_in_name='tag_dropout_mode',
                                special_tags_in_name='tag_dropout_special_tags', special_tag_mode_in_name='tag_dropout_special_tags_mode', delimiter_in_name='tag_delimiter',
                                keep_tags_count_in_name='keep_tags_count', text_out_name='prompt', regex_enabled_in_name='tag_dropout_special_tags_regex')
            caps_randomize = CapitalizeTags(text_in_name='prompt', enabled_in_name='caps_randomize_enable', probability_in_name='caps_randomize_probability',
                                            capitalize_mode_in_name='caps_randomize_mode', delimiter_in_name='tag_delimiter', convert_lowercase_in_name='caps_randomize_lowercase', text_out_name='prompt')
            shuffle_tags = ShuffleTags(text_in_name='prompt', enabled_in_name='enable_tag_shuffling', delimiter_in_name='tag_delimiter', keep_tags_count_in_name='keep_tags_count', text_out_name='prompt')
            output_module = OutputPipelineModule(['image', 'mask', 'prompt'])

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
                drop_tags,
                caps_randomize,
                shuffle_tags,
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
            prompt_output = data['prompt']

        filename_output = os.path.basename(preview_image_path)

        mask_tensor = torch.clamp(mask_tensor, 0.3, 1)
        image_tensor = image_tensor * mask_tensor

        image = functional.to_pil_image(image_tensor)

        image.thumbnail((300, 300))

        return image, filename_output, prompt_output

    def __update_concept_stats(self):
        #file size
        self.file_size_preview.configure(text=str(int(self.concept.concept_stats["file_size"]/1048576)) + " MB")
        self.processing_time.configure(text=str(round(self.concept.concept_stats["processing_time"], 2)) + " s")

        #directory count
        self.dir_count_preview.configure(text=self.concept.concept_stats["directory_count"])

        #image count
        self.image_count_preview.configure(text=self.concept.concept_stats["image_count"])
        self.image_count_mask_preview.configure(text=self.concept.concept_stats["image_with_mask_count"])
        self.image_count_caption_preview.configure(text=self.concept.concept_stats["image_with_caption_count"])

        #video count
        self.video_count_preview.configure(text=self.concept.concept_stats["video_count"])
        #self.video_count_mask_preview.configure(text=self.concept.concept_stats["video_with_mask_count"])
        self.video_count_caption_preview.configure(text=self.concept.concept_stats["video_with_caption_count"])

        #mask count
        self.mask_count_preview.configure(text=self.concept.concept_stats["mask_count"])
        self.mask_count_preview_unpaired.configure(text=self.concept.concept_stats["unpaired_masks"])

        #caption count
        if self.concept.concept_stats["subcaption_count"] > 0:
            self.caption_count_preview.configure(text=f'{self.concept.concept_stats["caption_count"]} ({self.concept.concept_stats["subcaption_count"]})')
        else:
            self.caption_count_preview.configure(text=self.concept.concept_stats["caption_count"])
        self.caption_count_preview_unpaired.configure(text=self.concept.concept_stats["unpaired_captions"])

        #resolution info
        max_pixels = self.concept.concept_stats["max_pixels"]
        avg_pixels = self.concept.concept_stats["avg_pixels"]
        min_pixels = self.concept.concept_stats["min_pixels"]

        if any(isinstance(x, str) for x in [max_pixels, avg_pixels, min_pixels]) or self.concept.concept_stats["image_count"] == 0:   #will be str if adv stats were not taken
            self.pixel_max_preview.configure(text="-")
            self.pixel_avg_preview.configure(text="-")
            self.pixel_min_preview.configure(text="-")
        else:
            #formatted as (#pixels/1000000) MP, width x height, \n filename
            self.pixel_max_preview.configure(text=f'{str(round(max_pixels[0]/1000000, 2))} MP, {max_pixels[2]}\n{max_pixels[1]}')
            self.pixel_avg_preview.configure(text=f'{str(round(avg_pixels/1000000, 2))} MP, ~{int(math.sqrt(avg_pixels))}w x {int(math.sqrt(avg_pixels))}h')
            self.pixel_min_preview.configure(text=f'{str(round(min_pixels[0]/1000000, 2))} MP, {min_pixels[2]}\n{min_pixels[1]}')

        #video length and fps info
        max_length = self.concept.concept_stats["max_length"]
        avg_length = self.concept.concept_stats["avg_length"]
        min_length = self.concept.concept_stats["min_length"]
        max_fps = self.concept.concept_stats["max_fps"]
        avg_fps = self.concept.concept_stats["avg_fps"]
        min_fps = self.concept.concept_stats["min_fps"]

        if any(isinstance(x, str) for x in [max_length, avg_length, min_length]) or self.concept.concept_stats["video_count"] == 0:   #will be str if adv stats were not taken
            self.length_max_preview.configure(text="-")
            self.length_avg_preview.configure(text="-")
            self.length_min_preview.configure(text="-")
            self.fps_max_preview.configure(text="-")
            self.fps_avg_preview.configure(text="-")
            self.fps_min_preview.configure(text="-")
        else:
            #formatted as (#frames) frames \n filename
            self.length_max_preview.configure(text=f'{int(max_length[0])} frames\n{max_length[1]}')
            self.length_avg_preview.configure(text=f'{int(avg_length)} frames')
            self.length_min_preview.configure(text=f'{int(min_length[0])} frames\n{min_length[1]}')
            #formatted as (#fps) fps \n filename
            self.fps_max_preview.configure(text=f'{int(max_fps[0])} fps\n{max_fps[1]}')
            self.fps_avg_preview.configure(text=f'{int(avg_fps)} fps')
            self.fps_min_preview.configure(text=f'{int(min_fps[0])} fps\n{min_fps[1]}')

        #caption info
        max_caption_length = self.concept.concept_stats["max_caption_length"]
        avg_caption_length = self.concept.concept_stats["avg_caption_length"]
        min_caption_length = self.concept.concept_stats["min_caption_length"]

        if any(isinstance(x, str) for x in [max_caption_length, avg_caption_length, min_caption_length]) or self.concept.concept_stats["caption_count"] == 0:   #will be str if adv stats were not taken
            self.caption_max_preview.configure(text="-")
            self.caption_avg_preview.configure(text="-")
            self.caption_min_preview.configure(text="-")
        else:
            #formatted as (#chars) chars, (#words) words, \n filename
            self.caption_max_preview.configure(text=f'{max_caption_length[0]} chars, {max_caption_length[2]} words\n{max_caption_length[1]}')
            self.caption_avg_preview.configure(text=f'{int(avg_caption_length[0])} chars, {int(avg_caption_length[1])} words')
            self.caption_min_preview.configure(text=f'{min_caption_length[0]} chars, {min_caption_length[2]} words\n{min_caption_length[1]}')

        #aspect bucketing
        aspect_buckets = self.concept.concept_stats["aspect_buckets"]
        if len(aspect_buckets) != 0 and max(val for val in aspect_buckets.values()) > 0:    #check aspect_bucket data exists and is not all zero
            min_val = min(val for val in aspect_buckets.values() if val > 0)                #smallest nonzero values
            if max(val for val in aspect_buckets.values()) > min_val:                       #check if any buckets larger than min_val exist - if all images are same aspect then there won't be
                min_val2 = min(val for val in aspect_buckets.values() if (val > 0 and val != min_val))  #second smallest bucket
            else:
                min_val2 = min_val  #if no second smallest bucket exists set to min_val
            min_aspect_buckets = {key: val for key,val in aspect_buckets.items() if val in (min_val, min_val2)}
            min_bucket_str = ""
            for key, val in min_aspect_buckets.items():
                min_bucket_str += f'aspect {self.decimal_to_aspect_ratio(key)} : {val} img\n'
            min_bucket_str.strip()
            self.small_bucket_preview.configure(text=min_bucket_str)

        self.bucket_ax.cla()
        aspects = [str(x) for x in list(aspect_buckets.keys())]
        aspect_ratios = [self.decimal_to_aspect_ratio(x) for x in list(aspect_buckets.keys())]
        counts = list(aspect_buckets.values())
        b = self.bucket_ax.bar(aspect_ratios, counts)
        self.bucket_ax.bar_label(b, color=self.text_color)
        sec = self.bucket_ax.secondary_xaxis(location=-0.1)
        sec.spines["bottom"].set_linewidth(0)
        sec.set_xticks([0, (len(aspects)-1)/2, len(aspects)-1], labels=["Wide", "Square", "Tall"])
        sec.tick_params('x', length=0)
        self.canvas.draw()

    def decimal_to_aspect_ratio(self, value : float):
        #find closest fraction to decimal aspect value and convert to a:b format
        aspect_fraction = fractions.Fraction(value).limit_denominator(16)
        aspect_string = f'{aspect_fraction.denominator}:{aspect_fraction.numerator}'
        return aspect_string

    def __get_concept_stats(self, advanced_checks: bool, wait_time: float):
        if not os.path.isdir(self.concept.path):
            print(f"Unable to get statistics for invalid concept path: {self.concept.path}")
            return
        start_time = time.perf_counter()
        last_update = time.perf_counter()
        self.cancel_scan_flag.clear()
        self.concept_stats_tab.after(0, self.__disable_scan_buttons)
        concept_path = self.get_concept_path(self.concept.path)

        if not concept_path:
           print(f"Unable to get statistics for invalid concept path: {self.concept.path}")
           self.concept_stats_tab.after(0, self.__enable_scan_buttons)
           return
        subfolders = [concept_path]

        stats_dict = concept_stats.init_concept_stats(advanced_checks)
        for path in subfolders:
            if self.cancel_scan_flag.is_set() or time.perf_counter() - start_time > wait_time:
                break
            stats_dict = concept_stats.folder_scan(path, stats_dict, advanced_checks, self.concept, start_time, wait_time, self.cancel_scan_flag)
            if self.concept.include_subdirectories and not self.cancel_scan_flag.is_set():     #add all subfolders of current directory to for loop
                subfolders.extend([f for f in os.scandir(path) if f.is_dir()])
            self.concept.concept_stats = stats_dict
            #update GUI approx every half second
            if time.perf_counter() > (last_update + 0.5):
                last_update = time.perf_counter()
                self.concept_stats_tab.after(0, self.__update_concept_stats)

        self.cancel_scan_flag.clear()
        self.concept_stats_tab.after(0, self.__enable_scan_buttons)
        self.concept_stats_tab.after(0, self.__update_concept_stats)

    def __get_concept_stats_threaded(self, advanced_checks : bool, waittime : float):
        self.scan_thread = threading.Thread(target=self.__get_concept_stats, args=[advanced_checks, waittime], daemon=True)
        self.scan_thread.start()

    def __disable_scan_buttons(self):
        self.refresh_basic_stats_button.configure(state="disabled")
        self.refresh_advanced_stats_button.configure(state="disabled")

    def __enable_scan_buttons(self):
        self.refresh_basic_stats_button.configure(state="normal")
        self.refresh_advanced_stats_button.configure(state="normal")

    def __cancel_concept_stats(self):
        self.cancel_scan_flag.set()

    def __auto_update_concept_stats(self):
        try:
            self.__update_concept_stats()      #load stats from config if available, else raises KeyError
            if self.concept.concept_stats["file_size"] == 0:  #force rescan if empty
                raise KeyError
        except KeyError:
            concept_path = self.get_concept_path(self.concept.path)
            if concept_path:
                self.__get_concept_stats(False, 2)    #force rescan if config is empty, timeout of 2 sec
                if self.concept.concept_stats["processing_time"] < 0.1:
                    self.__get_concept_stats(True, 2)    #do advanced scan automatically if basic took <0.1s

    def __ok(self):
        self.destroy()
