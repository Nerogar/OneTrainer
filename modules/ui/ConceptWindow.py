import asyncio
import math
import os
import pathlib
import random
import time

from modules.util import path_util
from modules.util.config.ConceptConfig import ConceptConfig
from modules.util.enum.BalancingStrategy import BalancingStrategy
from modules.util.ui import components
from modules.util.ui.UIState import UIState
from scripts import concept_stats

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
        self.geometry("800x700")
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
        self.concept_stats_tab = self.__concept_stats_tab(tabview.add("statistics"))

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
        self.caption_preview = ctk.CTkTextbox(master=frame, width = 300, height = 150, wrap="word", border_width=2)
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

    def __concept_stats_tab(self, master):
        frame = ctk.CTkScrollableFrame(master, fg_color="transparent")
        frame.grid_columnconfigure(0, weight=0, minsize=150)
        frame.grid_columnconfigure(1, weight=0, minsize=150)
        frame.grid_columnconfigure(2, weight=0, minsize=150)
        frame.grid_columnconfigure(3, weight=0, minsize=150)

        self.cancel_scan_flag = False

        #file size
        self.file_size_label = components.label(frame, 1, 0, "Total Size", pad=0,
                         tooltip="Total size of all image, mask, and caption files")
        self.file_size_label.configure(font=ctk.CTkFont(underline=True))
        self.file_size_preview = components.label(frame, 2, 0, pad=0, text="-")

        #subdirectory count
        self.dir_count_label = components.label(frame, 1, 1, "Directories", pad=0,
                         tooltip="Total number of directories including and under (if 'include subdirectories' is enabled) main concept directory")
        self.dir_count_label.configure(font=ctk.CTkFont(underline=True))
        self.dir_count_preview = components.label(frame, 2, 1, pad=0, text="-")

        #basic img stats
        self.image_count_label = components.label(frame, 3, 0, "\nTotal Images", pad=0,
                         tooltip="Total number of image files, any of the extensions " + str(path_util.SUPPORTED_IMAGE_EXTENSIONS) + ", excluding '-masklabel.png'")
        self.image_count_label.configure(font=ctk.CTkFont(underline=True))
        self.image_count_preview = components.label(frame, 4, 0, pad=0, text="-")
        self.mask_count_label = components.label(frame, 3, 1, "\nTotal Masks", pad=0,
                         tooltip="Total number of mask files, any file ending in '-masklabel.png'")
        self.mask_count_label.configure(font=ctk.CTkFont(underline=True))
        self.mask_count_preview = components.label(frame, 4, 1, pad=0, text="-")
        self.caption_count_label = components.label(frame, 3, 2, "\nTotal Captions", pad=0,
                         tooltip="Total number of caption files, any .txt file")
        self.caption_count_label.configure(font=ctk.CTkFont(underline=True))
        self.caption_count_preview = components.label(frame, 4, 2, pad=0, text="-")

        #advanced img stats
        self.image_count_mask_label = components.label(frame, 5, 0, "\nImages with Masks", pad=0,
                         tooltip="Total number of image files with an associated mask")
        self.image_count_mask_label.configure(font=ctk.CTkFont(underline=True))
        self.image_count_mask_preview = components.label(frame, 6, 0, pad=0, text="-")
        self.mask_count_label_unpaired = components.label(frame, 5, 1, "\nUnpaired Masks", pad=0,
                         tooltip="Total number of mask files which lack a corresponding image file - if >0, check your data set!")
        self.mask_count_label_unpaired.configure(font=ctk.CTkFont(underline=True))
        self.mask_count_preview_unpaired = components.label(frame, 6, 1, pad=0, text="-")
        self.image_count_caption_label = components.label(frame, 5, 2, "\nImages with Captions", pad=0,
                         tooltip="Total number of image files with an associated caption")
        self.image_count_caption_label.configure(font=ctk.CTkFont(underline=True))
        self.image_count_caption_preview = components.label(frame, 6, 2, pad=0, text="-")
        self.caption_count_label_unpaired = components.label(frame, 5, 3, "\nUnpaired Captions", pad=0,
                         tooltip="Total number of caption files which lack a corresponding image file - if >0, check your data set!")
        self.caption_count_label_unpaired.configure(font=ctk.CTkFont(underline=True))
        self.caption_count_preview_unpaired = components.label(frame, 6, 3, pad=0, text="-")

        #resolution info
        self.pixel_max_label = components.label(frame, 7, 0, "\nMax Pixels", pad=0,
                         tooltip="Largest image in the concept by number of pixels (width * height)")
        self.pixel_max_label.configure(font=ctk.CTkFont(underline=True))
        self.pixel_max_preview = components.label(frame, 8, 0, pad=0, text="-", wraplength=150)
        self.pixel_avg_label = components.label(frame, 7, 1, "\nAvg Pixels", pad=0,
                         tooltip="Average size of images in the concept by number of pixels (width * height)")
        self.pixel_avg_label.configure(font=ctk.CTkFont(underline=True))
        self.pixel_avg_preview = components.label(frame, 8, 1, pad=0, text="-", wraplength=150)
        self.pixel_min_label = components.label(frame, 7, 2, "\nMin Pixels", pad=0,
                         tooltip="Smallest image in the concept by number of pixels (width * height)")
        self.pixel_min_label.configure(font=ctk.CTkFont(underline=True))
        self.pixel_min_preview = components.label(frame, 8, 2, pad=0, text="-", wraplength=150)

        #caption info
        self.caption_max_label = components.label(frame, 9, 0, "\nMax Caption Length", pad=0,
                         tooltip="Largest caption in concept by character count")
        self.caption_max_label.configure(font=ctk.CTkFont(underline=True))
        self.caption_max_preview = components.label(frame, 10, 0, pad=0, text="-", wraplength=150)
        self.caption_avg_label = components.label(frame, 9, 1, "\nAvg Caption Length", pad=0,
                         tooltip="Average length of caption in concept by character count")
        self.caption_avg_label.configure(font=ctk.CTkFont(underline=True))
        self.caption_avg_preview = components.label(frame, 10, 1, pad=0, text="-", wraplength=150)
        self.caption_min_label = components.label(frame, 9, 2, "\nMin Caption Length", pad=0,
                         tooltip="Smallest caption in concept by character count")
        self.caption_min_label.configure(font=ctk.CTkFont(underline=True))
        self.caption_min_preview = components.label(frame, 10, 2, pad=0, text="-", wraplength=150)

        #bucket info
        self.aspect_bucket_label = components.label(frame, 11, 0, "\nAspect Bucketing", pad=0,
                         tooltip="List of all possible buckets and the number of images in each one, defined as width/height. Buckets range from 0.25 (1:4 extremely tall) to 4 (4:1 extremely wide).")
        self.aspect_bucket_label.configure(font=ctk.CTkFont(underline=True))
        self.small_bucket_label = components.label(frame, 11, 1, "\nSmallest Buckets", pad=0,
                         tooltip="Image buckets with the least nonzero total images - if 'batch size' is larger than this, these images will be dropped during training!")
        self.small_bucket_label.configure(font=ctk.CTkFont(underline=True))
        self.small_bucket_preview = components.label(frame, 12, 1, pad=0, text="-")

        # plot
        appearance_mode = AppearanceModeTracker.get_mode()
        background_color = self.winfo_rgb(ThemeManager.theme["CTkToplevel"]["fg_color"][appearance_mode])
        text_color = self.winfo_rgb(ThemeManager.theme["CTkLabel"]["text_color"][appearance_mode])
        background_color = f"#{int(background_color[0]/256):x}{int(background_color[1]/256):x}{int(background_color[2]/256):x}"
        text_color = f"#{int(text_color[0]/256):x}{int(text_color[1]/256):x}{int(text_color[2]/256):x}"

        plt.set_loglevel('WARNING')     #suppress errors about data type in bar chart
        self.bucket_fig, self.bucket_ax = plt.subplots(figsize=(7,2))
        self.canvas = FigureCanvasTkAgg(self.bucket_fig, master=frame)
        self.canvas.get_tk_widget().grid(row=13, column=0, columnspan=4, rowspan=2)
        self.bucket_fig.tight_layout()

        self.bucket_fig.set_facecolor(background_color)
        self.bucket_ax.set_facecolor(background_color)
        self.bucket_ax.spines['bottom'].set_color(text_color)
        self.bucket_ax.spines['left'].set_color(text_color)
        self.bucket_ax.spines['top'].set_visible(False)
        self.bucket_ax.spines['right'].set_color(text_color)
        self.bucket_ax.tick_params(axis='x', colors=text_color, which="both")
        self.bucket_ax.tick_params(axis='y', colors=text_color, which="both")
        self.bucket_ax.xaxis.label.set_color(text_color)
        self.bucket_ax.yaxis.label.set_color(text_color)

        #refresh stats - must be after all labels are defined or will give error
        components.button(master=frame, row=0, column=0, text="Refresh Basic", command=lambda: asyncio.run(self.__get_concept_stats(False, 9999)),
                          tooltip="Reload basic statistics for the concept directory")
        components.button(master=frame, row=0, column=1, text="Refresh Advanced", command=lambda: [asyncio.run(self.__get_concept_stats(False, 9999)), asyncio.run(self.__get_concept_stats(True, 9999))],
                          tooltip="Reload advanced statistics for the concept directory")       #run "basic" scan first before "advanced", seems to help the system cache the directories and run faster
        components.button(master=frame, row=0, column=2, text="Abort Scan", command=asyncio.run(self.__cancel_concept_stats()),
                          tooltip="Stop the currently running scan if it's taking a long time - will be slow on large folders and on HDDs")
        self.processing_time = components.label(frame, 0, 3, text="-", tooltip="Time taken to process concept directory")

        #automatically get basic stats if available
        try:
            self.__update_concept_stats()      #load stats from config if available
        except KeyError:
            start_time = time.perf_counter()
            asyncio.run(self.__get_concept_stats(False, 1))    #force rescan if config is empty, timeout of 1 sec
            if (time.perf_counter() - start_time) < 0.1:
                asyncio.run(self.__get_concept_stats(True, 1))    #do advanced scan automatically if basic <0.1s

        except FileNotFoundError:              #avoid error when loading concept window without config path defined
            pass

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

        #mask count
        self.mask_count_preview.configure(text=self.concept.concept_stats["mask_count"])
        self.mask_count_preview_unpaired.configure(text=self.concept.concept_stats["unpaired_masks"])

        #caption count
        self.caption_count_preview.configure(text=self.concept.concept_stats["caption_count"])
        self.caption_count_preview_unpaired.configure(text=self.concept.concept_stats["unpaired_captions"])

        #resolution info
        max_pixels = self.concept.concept_stats["max_pixels"]
        avg_pixels = self.concept.concept_stats["avg_pixels"]
        min_pixels = self.concept.concept_stats["min_pixels"]

        if any(isinstance(x, str) for x in [max_pixels, avg_pixels, min_pixels]):   #will be str if adv stats were not taken
            self.pixel_max_preview.configure(text=max_pixels)
            self.pixel_avg_preview.configure(text=avg_pixels)
            self.pixel_min_preview.configure(text=min_pixels)
        else:
            #formatted as (#pixels/1000000) MP, widthxheight, \n filename
            self.pixel_max_preview.configure(text=f'{str(round(max_pixels[0]/1000000, 2))} MP, {max_pixels[2]}\n{max_pixels[1]}')
            self.pixel_avg_preview.configure(text=f'{str(round(avg_pixels/1000000, 2))} MP, ~{int(math.sqrt(avg_pixels))}x{int(math.sqrt(avg_pixels))}')
            self.pixel_min_preview.configure(text=f'{str(round(min_pixels[0]/1000000, 2))} MP, {min_pixels[2]}\n{min_pixels[1]}')

        #caption info
        max_caption_length = self.concept.concept_stats["max_caption_length"]
        avg_caption_length = self.concept.concept_stats["avg_caption_length"]
        min_caption_length = self.concept.concept_stats["min_caption_length"]

        if any(isinstance(x, str) for x in [max_caption_length, avg_caption_length, min_caption_length]):   #will be str if adv stats were not taken
            self.caption_max_preview.configure(text=max_caption_length)
            self.caption_avg_preview.configure(text=avg_caption_length)
            self.caption_min_preview.configure(text=min_caption_length)
        else:
            #formatted as (#pixels/1000000) MP, widthxheight, \n filename
            self.caption_max_preview.configure(text=f'{max_caption_length[0]} chars, {max_caption_length[2]} words\n{max_caption_length[1]}')
            self.caption_avg_preview.configure(text=f'{int(avg_caption_length[0])} chars, {int(avg_caption_length[1])} words')
            self.caption_min_preview.configure(text=f'{min_caption_length[0]} chars, {min_caption_length[2]} words\n{min_caption_length[1]}')

        #bucketing
        aspect_buckets = self.concept.concept_stats["aspect_buckets"]
        if len(aspect_buckets) != 0 and max(val for val in aspect_buckets.values()) > 0:    #check aspect_bucket data exists and is not all zero
            min_val = min(val for val in aspect_buckets.values() if val > 0)                #smallest nonzero values
            if max(val for val in aspect_buckets.values()) > min_val:                       #check if any buckets larger than min_val exist
                min_val2 = min(val for val in aspect_buckets.values() if (val > 0 and val != min_val))  #second smallest bucket
            else:
                min_val2 = min_val  #if no second smallest bucket exists set to min_val
            min_aspect_buckets = {key: val for key,val in aspect_buckets.items() if val in (min_val, min_val2)}
            min_bucket_str = ""
            for key, val in min_aspect_buckets.items():
                min_bucket_str += f'aspect {key}: {val} img\n'
            min_bucket_str.strip()
            self.small_bucket_preview.configure(text=min_bucket_str)

        self.bucket_ax.cla()
        aspects = [str(x) for x in list(aspect_buckets.keys())]
        counts = list(aspect_buckets.values())
        b = self.bucket_ax.bar(aspects, counts)
        self.bucket_ax.bar_label(b)
        self.canvas.draw()

    async def __get_concept_stats(self, advanced_checks : bool, waittime : float):
        start_time = time.perf_counter()
        last_update = time.perf_counter()
        subfolders = [self.concept.path]
        stats_dict = concept_stats.init_concept_stats(self.concept, advanced_checks)
        for dir in subfolders:
            stats_dict = concept_stats.folder_scan(dir, stats_dict, advanced_checks, self.concept)
            stats_dict["processing_time"] = time.perf_counter() - start_time
            if self.concept.include_subdirectories:
                subfolders.extend([f for f in os.scandir(dir) if f.is_dir()])
            self.concept.concept_stats = stats_dict
            #cancel if longer than waiting time
            if (time.perf_counter() - start_time) > waittime or self.cancel_scan_flag:
                stats_dict = concept_stats.init_concept_stats(self.concept, advanced_checks)
                stats_dict["processing_time"] = time.perf_counter() - start_time
                self.concept.concept_stats = stats_dict
                self.cancel_scan_flag = False
                break
            #update GUI approx every second
            if time.perf_counter() > (last_update + 1):
                last_update = time.perf_counter()
                self.__update_concept_stats()
                self.concept_stats_tab.update()

        self.__update_concept_stats()

    async def __cancel_concept_stats(self):
        self.cancel_scan_flag = True

    async def __get_concept_stats_threaded(self, advanced_checks : bool, waittime : float):
        await asyncio.to_thread(self.__get_concept_stats(advanced_checks, waittime))

    def __ok(self):
        self.destroy()
