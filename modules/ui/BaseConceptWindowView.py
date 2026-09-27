import fractions
import math

from modules.util import path_util
from modules.util.enum.BalancingStrategy import BalancingStrategy
from modules.util.enum.ConceptType import ConceptType


class BaseConceptWindowView:
    def __init__(self, components):
        self.components = components
        self.bucket_ax = None
        self.text_color = None
        self.canvas = None

    def build_general_tab(self, frame, controller, ui_state, text_ui_state):
        # name
        self.components.label(frame, 0, 0, "Name",
                         tooltip="Name of the concept")
        self.components.entry(frame, 0, 1, ui_state, "name")

        # enabled
        self.components.label(frame, 1, 0, "Enabled",
                         tooltip="Enable or disable this concept")
        self.components.switch(frame, 1, 1, ui_state, "enabled")

        # concept type
        self.components.label(frame, 2, 0, "Concept Type",
                         tooltip="STANDARD: Standard finetuning with the sample as training target\n"
                                 "VALIDATION: Use concept for validation instead of training\n"
                                 "PRIOR_PREDICTION: Use the sample to make a prediction using the model as it was before training. This prediction is then used as the training target "
                                 "for the model in training. This can be used as regularisation and to preserve prior model knowledge while finetuning the model on other concepts. "
                                 "Only implemented for LoRA.",
                         wide_tooltip=True)
        self.components.options(frame, 2, 1, [str(x) for x in list(ConceptType)], ui_state, "type")

        # path
        self.components.label(frame, 3, 0, "Path",
                         tooltip="Path where the training data is located")
        self.components.path_entry(frame, 3, 1, ui_state, "path", mode="dir")
        self.components.button(frame, 3, 2, text="download now", command=controller.download_dataset_threaded,
                          tooltip="Download dataset from Huggingface now, for the purpose of previewing and statistics. Otherwise, it will be downloaded when you start training. Path must be a Huggingface repository.")

        # prompt source
        self.components.label(frame, 4, 0, "Prompt Source",
                         tooltip="The source for prompts used during training. When selecting \"From single text file\", select a text file that contains a list of prompts")
        prompt_path_entry = self.components.path_entry(frame, 4, 2, text_ui_state, "prompt_path", mode="file")

        def set_prompt_path_entry_enabled(option: str):
            self.components.set_widget_enabled(prompt_path_entry, option == 'concept')

        self.components.options_kv(frame, 4, 1, [
            ("From text file per sample", 'sample'),
            ("From single text file", 'concept'),
            ("From image file name", 'filename'),
        ], text_ui_state, "prompt_source", command=set_prompt_path_entry_enabled)
        set_prompt_path_entry_enabled(controller.concept.text.prompt_source)

        # include subdirectories
        self.components.label(frame, 5, 0, "Include Subdirectories",
                         tooltip="Includes images from subdirectories into the dataset")
        self.components.switch(frame, 5, 1, ui_state, "include_subdirectories")

        # image variations
        self.components.label(frame, 6, 0, "Image Variations",
                         tooltip="The number of different image versions to cache if latent caching is enabled.")
        self.components.entry(frame, 6, 1, ui_state, "image_variations")

        # text variations
        self.components.label(frame, 7, 0, "Text Variations",
                         tooltip="The number of different text versions to cache if latent caching is enabled.")
        self.components.entry(frame, 7, 1, ui_state, "text_variations")

        # balancing
        self.components.label(frame, 8, 0, "Balancing",
                         tooltip="The number of samples used during training. Use repeats to multiply the concept, or samples to specify an exact number of samples used in each epoch.")
        self.components.entry(frame, 8, 1, ui_state, "balancing")
        self.components.options(frame, 8, 2, [str(x) for x in list(BalancingStrategy)], ui_state, "balancing_strategy")

        # loss weight
        self.components.label(frame, 9, 0, "Loss Weight",
                         tooltip="The loss multiplyer for this concept.")
        self.components.entry(frame, 9, 1, ui_state, "loss_weight")

    def build_image_augmentation_tab(self, frame, controller, image_ui_state):
        # header
        self.components.label(frame, 0, 1, "Random",
                         tooltip="Enable this augmentation with random values")
        self.components.label(frame, 0, 2, "Fixed",
                         tooltip="Enable this augmentation with fixed values")

        # crop jitter
        self.components.label(frame, 1, 0, "Crop Jitter",
                         tooltip="Enables random cropping of samples")
        self.components.switch(frame, 1, 1, image_ui_state, "enable_crop_jitter")

        # random flip
        self.components.label(frame, 2, 0, "Random Flip",
                         tooltip="Randomly flip the sample during training")
        self.components.switch(frame, 2, 1, image_ui_state, "enable_random_flip")
        self.components.switch(frame, 2, 2, image_ui_state, "enable_fixed_flip")

        # random rotation
        self.components.label(frame, 3, 0, "Random Rotation",
                         tooltip="Randomly rotates the sample during training")
        self.components.switch(frame, 3, 1, image_ui_state, "enable_random_rotate")
        self.components.switch(frame, 3, 2, image_ui_state, "enable_fixed_rotate")
        self.components.entry(frame, 3, 3, image_ui_state, "random_rotate_max_angle")

        # random brightness
        self.components.label(frame, 4, 0, "Random Brightness",
                         tooltip="Randomly adjusts the brightness of the sample during training")
        self.components.switch(frame, 4, 1, image_ui_state, "enable_random_brightness")
        self.components.switch(frame, 4, 2, image_ui_state, "enable_fixed_brightness")
        self.components.entry(frame, 4, 3, image_ui_state, "random_brightness_max_strength")

        # random contrast
        self.components.label(frame, 5, 0, "Random Contrast",
                         tooltip="Randomly adjusts the contrast of the sample during training")
        self.components.switch(frame, 5, 1, image_ui_state, "enable_random_contrast")
        self.components.switch(frame, 5, 2, image_ui_state, "enable_fixed_contrast")
        self.components.entry(frame, 5, 3, image_ui_state, "random_contrast_max_strength")

        # random saturation
        self.components.label(frame, 6, 0, "Random Saturation",
                         tooltip="Randomly adjusts the saturation of the sample during training")
        self.components.switch(frame, 6, 1, image_ui_state, "enable_random_saturation")
        self.components.switch(frame, 6, 2, image_ui_state, "enable_fixed_saturation")
        self.components.entry(frame, 6, 3, image_ui_state, "random_saturation_max_strength")

        # random hue
        self.components.label(frame, 7, 0, "Random Hue",
                         tooltip="Randomly adjusts the hue of the sample during training")
        self.components.switch(frame, 7, 1, image_ui_state, "enable_random_hue")
        self.components.switch(frame, 7, 2, image_ui_state, "enable_fixed_hue")
        self.components.entry(frame, 7, 3, image_ui_state, "random_hue_max_strength")

        # random circular mask shrink
        self.components.label(frame, 8, 0, "Circular Mask Generation",
                         tooltip="Automatically create circular masks for masked training")
        self.components.switch(frame, 8, 1, image_ui_state, "enable_random_circular_mask_shrink")

        # random rotate and crop
        self.components.label(frame, 9, 0, "Random Rotate and Crop",
                         tooltip="Randomly rotate the training samples and crop to the masked region")
        self.components.switch(frame, 9, 1, image_ui_state, "enable_random_mask_rotate_crop")

        # circular mask generation
        self.components.label(frame, 10, 0, "Resolution Override",
                         tooltip="Override the resolution for this concept. Optionally specify multiple resolutions separated by a comma, or a single exact resolution in the format <width>x<height>")
        self.components.switch(frame, 10, 2, image_ui_state, "enable_resolution_override")
        self.components.entry(frame, 10, 3, image_ui_state, "resolution_override")

    def build_text_augmentation_tab(self, frame, controller, text_ui_state):
        # tag shuffling
        self.components.label(frame, 0, 0, "Tag Shuffling",
                         tooltip="Enables tag shuffling")
        self.components.switch(frame, 0, 1, text_ui_state, "enable_tag_shuffling")

        # keep tag count
        self.components.label(frame, 1, 0, "Tag Delimiter",
                         tooltip="The delimiter between tags")
        self.components.entry(frame, 1, 1, text_ui_state, "tag_delimiter")

        # keep tag count
        self.components.label(frame, 2, 0, "Keep Tag Count",
                         tooltip="The number of tags at the start of the caption that are not shuffled or dropped")
        self.components.entry(frame, 2, 1, text_ui_state, "keep_tags_count")

        # tag dropout
        self.components.label(frame, 3, 0, "Tag Dropout",
                         tooltip="Enables random dropout for tags in the captions.")
        self.components.switch(frame, 3, 1, text_ui_state, "tag_dropout_enable")
        self.components.label(frame, 4, 0, "Dropout Mode",
                         tooltip="Method used to drop captions. 'Full' will drop the entire caption past the 'kept' tags with a certain probability, 'Random' will drop individual tags with the set probability, and 'Random Weighted' will linearly increase the probability of dropping tags, more likely to preseve tags near the front with full probability to drop at the end.")
        self.components.options_kv(frame, 4, 1, [
            ("Full", 'FULL'),
            ("Random", 'RANDOM'),
            ("Random Weighted", 'RANDOM WEIGHTED'),
        ], text_ui_state, "tag_dropout_mode", None)
        self.components.label(frame, 4, 2, "Probability",
                         tooltip="Probability to drop tags, from 0 to 1.")
        self.components.entry(frame, 4, 3, text_ui_state, "tag_dropout_probability")

        self.components.label(frame, 5, 0, "Special Dropout Tags",
                         tooltip="List of tags which will be whitelisted/blacklisted by dropout. 'Whitelist' tags will never be dropped but all others may be, 'Blacklist' tags may be dropped but all others will never be, 'None' may drop any tags. Can specify either a delimiter-separated list in the field, or a file path to a .txt or .csv file with entries separated by newlines.")
        self.components.options_kv(frame, 5, 1, [
            ("None", 'NONE'),
            ("Blacklist", 'BLACKLIST'),
            ("Whitelist", 'WHITELIST'),
        ], text_ui_state, "tag_dropout_special_tags_mode", None)
        self.components.entry(frame, 5, 2, text_ui_state, "tag_dropout_special_tags")
        self.components.label(frame, 6, 0, "Special Tags Regex",
                         tooltip="Interpret special tags with regex, such as 'photo.*' to match 'photo, photograph, photon' but not 'telephoto'. Includes exception for '/(' and '/)' syntax found in many booru/e6 tags.")
        self.components.switch(frame, 6, 1, text_ui_state, "tag_dropout_special_tags_regex")

        #capitalization randomization
        self.components.label(frame, 7, 0, "Randomize Capitalization",
                         tooltip="Enables randomization of capitalization for tags in the caption.")
        self.components.switch(frame, 7, 1, text_ui_state, "caps_randomize_enable")
        self.components.label(frame, 7, 2, "Force Lowercase",
                         tooltip="If enabled, converts the caption to lowercase before any further processing.")
        self.components.switch(frame, 7, 3, text_ui_state, "caps_randomize_lowercase")

        self.components.label(frame, 8, 0, "Captialization Mode",
                         tooltip="Comma-separated list of types of capitalization randomization to perform. 'capslock' for ALL CAPS, 'title' for First Letter Of Every Word, 'first' for First word only, 'random' for rAndOMiZeD lEtTERs.")
        self.components.entry(frame, 8, 1, text_ui_state, "caps_randomize_mode")
        self.components.label(frame, 8, 2, "Probability",
                         tooltip="Probability to randomize capitialization of each tag, from 0 to 1.")
        self.components.entry(frame, 8, 3, text_ui_state, "caps_randomize_probability")

    def build_concept_stats_tab(self, frame, controller):
        self.concept_stats_tab = frame

        #file size
        self.file_size_label = self.components.label(frame, 1, 0, "Total Size", pad=0,
                         tooltip="Total size of all image, mask, and caption files in MB", underline=True)
        self.file_size_preview = self.components.label(frame, 2, 0, pad=0, text="-")

        #subdirectory count
        self.dir_count_label = self.components.label(frame, 1, 1, "Directories", pad=0,
                         tooltip="Total number of directories including and under (if 'include subdirectories' is enabled) the main concept directory", underline=True)
        self.dir_count_preview = self.components.label(frame, 2, 1, pad=0, text="-")

        #basic img/vid stats - count of each type in the concept
        #the \n at the start of the label gives it better vertical spacing with other rows
        self.image_count_label = self.components.label(frame, 3, 0, "\nTotal Images", pad=0,
                         tooltip="Total number of image files, any of the extensions " + str(path_util.SUPPORTED_IMAGE_EXTENSIONS) + ", excluding '-masklabel.png and -condlabel.png'", underline=True)
        self.image_count_preview = self.components.label(frame, 4, 0, pad=0, text="-")
        self.video_count_label = self.components.label(frame, 3, 1, "\nTotal Videos", pad=0,
                         tooltip="Total number of video files, any of the extensions " + str(path_util.SUPPORTED_VIDEO_EXTENSIONS), underline=True)
        self.video_count_preview = self.components.label(frame, 4, 1, pad=0, text="-")
        self.mask_count_label = self.components.label(frame, 3, 2, "\nTotal Masks", pad=0,
                         tooltip="Total number of mask files, any file ending in '-masklabel.png'", underline=True)
        self.mask_count_preview = self.components.label(frame, 4, 2, pad=0, text="-")
        self.caption_count_label = self.components.label(frame, 3, 3, "\nTotal Captions", pad=0,
                         tooltip="Total number of caption files, any .txt file. With advanced scan, includes the total number of captions on separate lines across all files in parentheses.", underline=True)
        self.caption_count_preview = self.components.label(frame, 4, 3, pad=0, text="-")

        #advanced img/vid stats - how many img/vid files have a mask or caption of the same name
        self.image_count_mask_label = self.components.label(frame, 5, 0, "\nImages with Masks", pad=0,
                         tooltip="Total number of image files with an associated mask", underline=True)
        self.image_count_mask_preview = self.components.label(frame, 6, 0, pad=0, text="-")
        self.mask_count_label_unpaired = self.components.label(frame, 5, 1, "\nUnpaired Masks", pad=0,
                         tooltip="Total number of mask files which lack a corresponding image file - if >0, check your data set!", underline=True)
        self.mask_count_preview_unpaired = self.components.label(frame, 6, 1, pad=0, text="-")
        #currently no masks for videos?

        self.image_count_caption_label = self.components.label(frame, 7, 0, "\nImages with Captions", pad=0,
                         tooltip="Total number of image files with an associated caption", underline=True)
        self.image_count_caption_preview = self.components.label(frame, 8, 0, pad=0, text="-")
        self.video_count_caption_label = self.components.label(frame, 7, 1, "\nVideos with Captions", pad=0,
                         tooltip="Total number of video files with an associated caption", underline=True)
        self.video_count_caption_preview = self.components.label(frame, 8, 1, pad=0, text="-")
        self.caption_count_label_unpaired = self.components.label(frame, 7, 2, "\nUnpaired Captions", pad=0,
                         tooltip="Total number of caption files which lack a corresponding image file - if >0, check your data set! If using 'from file name' or 'from single text file' then this can be ignored.", underline=True)
        self.caption_count_preview_unpaired = self.components.label(frame, 8, 2, pad=0, text="-")

        #resolution info
        self.pixel_max_label = self.components.label(frame, 9, 0, "\nMax Pixels", pad=0,
                         tooltip="Largest image in the concept by number of pixels (width * height)", underline=True)
        self.pixel_max_preview = self.components.label(frame, 10, 0, pad=0, text="-", wraplength=150)
        self.pixel_avg_label = self.components.label(frame, 9, 1, "\nAvg Pixels", pad=0,
                         tooltip="Average size of images in the concept by number of pixels (width * height)", underline=True)
        self.pixel_avg_preview = self.components.label(frame, 10, 1, pad=0, text="-", wraplength=150)
        self.pixel_min_label = self.components.label(frame, 9, 2, "\nMin Pixels", pad=0,
                         tooltip="Smallest image in the concept by number of pixels (width * height)", underline=True)
        self.pixel_min_preview = self.components.label(frame, 10, 2, pad=0, text="-", wraplength=150)

        #video length info
        self.length_max_label = self.components.label(frame, 11, 0, "\nMax Length", pad=0,
                         tooltip="Longest video in the concept by number of frames", underline=True)
        self.length_max_preview = self.components.label(frame, 12, 0, pad=0, text="-", wraplength=150)
        self.length_avg_label = self.components.label(frame, 11, 1, "\nAvg Length", pad=0,
                         tooltip="Average length of videos in the concept by number of frames", underline=True)
        self.length_avg_preview = self.components.label(frame, 12, 1, pad=0, text="-", wraplength=150)
        self.length_min_label = self.components.label(frame, 11, 2, "\nMin Length", pad=0,
                         tooltip="Shortest video in the concept by number of frames", underline=True)
        self.length_min_preview = self.components.label(frame, 12, 2, pad=0, text="-", wraplength=150)

        #video fps info
        self.fps_max_label = self.components.label(frame, 13, 0, "\nMax FPS", pad=0,
                         tooltip="Video in concept with highest fps", underline=True)
        self.fps_max_preview = self.components.label(frame, 14, 0, pad=0, text="-", wraplength=150)
        self.fps_avg_label = self.components.label(frame, 13, 1, "\nAvg FPS", pad=0,
                         tooltip="Average fps of videos in the concept", underline=True)
        self.fps_avg_preview = self.components.label(frame, 14, 1, pad=0, text="-", wraplength=150)
        self.fps_min_label = self.components.label(frame, 13, 2, "\nMin FPS", pad=0,
                         tooltip="Video in concept with the lowest fps", underline=True)
        self.fps_min_preview = self.components.label(frame, 14, 2, pad=0, text="-", wraplength=150)

        #caption info
        self.caption_max_label = self.components.label(frame, 15, 0, "\nMax Caption Length", pad=0,
                         tooltip="Largest caption in concept by character count. For token count, assume ~2 tokens/word", underline=True)
        self.caption_max_preview = self.components.label(frame, 16, 0, pad=0, text="-", wraplength=150)
        self.caption_avg_label = self.components.label(frame, 15, 1, "\nAvg Caption Length", pad=0,
                         tooltip="Average length of caption in concept by character count. For token count, assume ~2 tokens/word", underline=True)
        self.caption_avg_preview = self.components.label(frame, 16, 1, pad=0, text="-", wraplength=150)
        self.caption_min_label = self.components.label(frame, 15, 2, "\nMin Caption Length", pad=0,
                         tooltip="Smallest caption in concept by character count. For token count, assume ~2 tokens/word", underline=True)
        self.caption_min_preview = self.components.label(frame, 16, 2, pad=0, text="-", wraplength=150)

        #aspect bucket info
        self.aspect_bucket_label = self.components.label(frame, 17, 0, "\nAspect Bucketing", pad=0,
                         tooltip="Graph of all possible buckets and the number of images in each one, defined as height/width. Buckets range from 0.25 (4:1 extremely wide) to 4 (1:4 extremely tall). \
                            Images which don't match a bucket exactly are cropped to the nearest one.", underline=True)
        self.small_bucket_label = self.components.label(frame, 17, 1, "\nSmallest Buckets", pad=0,
                         tooltip="Image buckets with the least nonzero total images - if 'batch size' is larger than this, these images will be ignored during training! See the wiki for more details.", underline=True)
        self.small_bucket_preview = self.components.label(frame, 18, 1, pad=0, text="-")

        #refresh stats - must be after all labels are defined or will give error
        self.refresh_basic_stats_button = self.components.button(master=frame, row=0, column=0, text="Refresh Basic", command=lambda: controller.get_concept_stats_threaded(self, False, 9999),
                          tooltip="Reload basic statistics for the concept directory")
        self.refresh_advanced_stats_button = self.components.button(master=frame, row=0, column=1, text="Refresh Advanced", command=lambda: controller.get_concept_stats_threaded(self, True, 9999),
                          tooltip="Reload advanced statistics for the concept directory")       #run "basic" scan first before "advanced", seems to help the system cache the directories and run faster
        self.cancel_stats_button = self.components.button(master=frame, row=0, column=2, text="Abort Scan", command=lambda: self._cancel_concept_stats(controller),
                          tooltip="Stop the currently running scan if it's taking a long time - advanced scan will be slow on large folders and on HDDs")
        self.processing_time = self.components.label(frame, 0, 3, text="-", tooltip="Time taken to process concept directory")

    def _update_concept_stats(self, controller):
        #file size
        self.components.set_label_text(self.file_size_preview, str(int(controller.concept.concept_stats["file_size"]/1048576)) + " MB")
        self.components.set_label_text(self.processing_time, str(round(controller.concept.concept_stats["processing_time"], 2)) + " s")

        #directory count
        self.components.set_label_text(self.dir_count_preview, controller.concept.concept_stats["directory_count"])

        #image count
        self.components.set_label_text(self.image_count_preview, controller.concept.concept_stats["image_count"])
        self.components.set_label_text(self.image_count_mask_preview, controller.concept.concept_stats["image_with_mask_count"])
        self.components.set_label_text(self.image_count_caption_preview, controller.concept.concept_stats["image_with_caption_count"])

        #video count
        self.components.set_label_text(self.video_count_preview, controller.concept.concept_stats["video_count"])
        #self.components.set_label_text(self.video_count_mask_preview, controller.concept.concept_stats["video_with_mask_count"])
        self.components.set_label_text(self.video_count_caption_preview, controller.concept.concept_stats["video_with_caption_count"])

        #mask count
        self.components.set_label_text(self.mask_count_preview, controller.concept.concept_stats["mask_count"])
        self.components.set_label_text(self.mask_count_preview_unpaired, controller.concept.concept_stats["unpaired_masks"])

        #caption count
        if controller.concept.concept_stats["subcaption_count"] > 0:
            self.components.set_label_text(self.caption_count_preview, f'{controller.concept.concept_stats["caption_count"]} ({controller.concept.concept_stats["subcaption_count"]})')
        else:
            self.components.set_label_text(self.caption_count_preview, controller.concept.concept_stats["caption_count"])
        self.components.set_label_text(self.caption_count_preview_unpaired, controller.concept.concept_stats["unpaired_captions"])

        #resolution info
        max_pixels = controller.concept.concept_stats["max_pixels"]
        avg_pixels = controller.concept.concept_stats["avg_pixels"]
        min_pixels = controller.concept.concept_stats["min_pixels"]

        if any(isinstance(x, str) for x in [max_pixels, avg_pixels, min_pixels]) or controller.concept.concept_stats["image_count"] == 0:   #will be str if adv stats were not taken
            self.components.set_label_text(self.pixel_max_preview, "-")
            self.components.set_label_text(self.pixel_avg_preview, "-")
            self.components.set_label_text(self.pixel_min_preview, "-")
        else:
            #formatted as (#pixels/1000000) MP, width x height, \n filename
            self.components.set_label_text(self.pixel_max_preview, f'{str(round(max_pixels[0]/1000000, 2))} MP, {max_pixels[2]}\n{max_pixels[1]}')
            self.components.set_label_text(self.pixel_avg_preview, f'{str(round(avg_pixels/1000000, 2))} MP, ~{int(math.sqrt(avg_pixels))}w x {int(math.sqrt(avg_pixels))}h')
            self.components.set_label_text(self.pixel_min_preview, f'{str(round(min_pixels[0]/1000000, 2))} MP, {min_pixels[2]}\n{min_pixels[1]}')

        #video length and fps info
        max_length = controller.concept.concept_stats["max_length"]
        avg_length = controller.concept.concept_stats["avg_length"]
        min_length = controller.concept.concept_stats["min_length"]
        max_fps = controller.concept.concept_stats["max_fps"]
        avg_fps = controller.concept.concept_stats["avg_fps"]
        min_fps = controller.concept.concept_stats["min_fps"]

        if any(isinstance(x, str) for x in [max_length, avg_length, min_length]) or controller.concept.concept_stats["video_count"] == 0:   #will be str if adv stats were not taken
            self.components.set_label_text(self.length_max_preview, "-")
            self.components.set_label_text(self.length_avg_preview, "-")
            self.components.set_label_text(self.length_min_preview, "-")
            self.components.set_label_text(self.fps_max_preview, "-")
            self.components.set_label_text(self.fps_avg_preview, "-")
            self.components.set_label_text(self.fps_min_preview, "-")
        else:
            #formatted as (#frames) frames \n filename
            self.components.set_label_text(self.length_max_preview, f'{int(max_length[0])} frames\n{max_length[1]}')
            self.components.set_label_text(self.length_avg_preview, f'{int(avg_length)} frames')
            self.components.set_label_text(self.length_min_preview, f'{int(min_length[0])} frames\n{min_length[1]}')
            #formatted as (#fps) fps \n filename
            self.components.set_label_text(self.fps_max_preview, f'{int(max_fps[0])} fps\n{max_fps[1]}')
            self.components.set_label_text(self.fps_avg_preview, f'{int(avg_fps)} fps')
            self.components.set_label_text(self.fps_min_preview, f'{int(min_fps[0])} fps\n{min_fps[1]}')

        #caption info
        max_caption_length = controller.concept.concept_stats["max_caption_length"]
        avg_caption_length = controller.concept.concept_stats["avg_caption_length"]
        min_caption_length = controller.concept.concept_stats["min_caption_length"]

        if any(isinstance(x, str) for x in [max_caption_length, avg_caption_length, min_caption_length]) or controller.concept.concept_stats["caption_count"] == 0:   #will be str if adv stats were not taken
            self.components.set_label_text(self.caption_max_preview, "-")
            self.components.set_label_text(self.caption_avg_preview, "-")
            self.components.set_label_text(self.caption_min_preview, "-")
        else:
            #formatted as (#chars) chars, (#words) words, \n filename
            self.components.set_label_text(self.caption_max_preview, f'{max_caption_length[0]} chars, {max_caption_length[2]} words\n{max_caption_length[1]}')
            self.components.set_label_text(self.caption_avg_preview, f'{int(avg_caption_length[0])} chars, {int(avg_caption_length[1])} words')
            self.components.set_label_text(self.caption_min_preview, f'{min_caption_length[0]} chars, {min_caption_length[2]} words\n{min_caption_length[1]}')

        #aspect bucketing
        aspect_buckets = controller.concept.concept_stats["aspect_buckets"]
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
            self.components.set_label_text(self.small_bucket_preview, min_bucket_str)

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

    def _disable_scan_buttons(self):
        self.components.set_widget_enabled(self.refresh_basic_stats_button, False)
        self.components.set_widget_enabled(self.refresh_advanced_stats_button, False)

    def _enable_scan_buttons(self):
        self.components.set_widget_enabled(self.refresh_basic_stats_button, True)
        self.components.set_widget_enabled(self.refresh_advanced_stats_button, True)

    def _cancel_concept_stats(self, controller):
        controller.cancel_scan_flag.set()
