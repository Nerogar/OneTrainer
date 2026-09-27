import webbrowser
from abc import ABC, abstractmethod


class BaseVideoToolUIView(ABC):
    def __init__(self, components):
        self.components = components

    def build_clip_extract_tab(self, frame, controller, ui_state):
        # single video
        self.components.label(frame, 0, 0, "Single Video",
                         tooltip="Link to single video file to process.")
        self.components.path_entry(frame, 0, 1, ui_state, "clip_single",
                               mode="file", allow_model_files=False, allow_video_files=True)
        self.components.button(frame, 0, 2, "Extract Single",
                          command=lambda: self._extract_clips(False, controller))

        # time range
        self.components.label(frame, 1, 0, "  Time Range",
                         tooltip="Time range to limit selection for single video, \
                            format as hour:minute:second, minute:second, or seconds.")
        self.components.entry(frame, 1, 1, ui_state, "clip_time_start", width=100, sticky="w")
        self.components.entry(frame, 1, 1, ui_state, "clip_time_end", width=100, sticky="e")

        # directory of videos
        self.components.label(frame, 2, 0, "Directory",
                         tooltip="Path to directory with multiple videos to process, including in subdirectories.")
        self.components.path_entry(frame, 2, 1, ui_state, "clip_list", mode="dir")
        self.components.button(frame, 2, 2, "Extract Directory",
                          command=lambda: self._extract_clips(True, controller))

        # output directory
        self.components.label(frame, 3, 0, "Output",
                         tooltip="Path to folder where extracted clips will be saved.")
        self.components.path_entry(frame, 3, 1, ui_state, "clip_output", mode="dir")

        # output to subdirectories
        self.components.label(frame, 4, 0, "Output to\nSubdirectories",
                         tooltip="If enabled, files are saved to subfolders based on filename and input directory. \
                            Otherwise will all be saved to the top level of the output directory.")
        self.components.switch(frame, 4, 1, ui_state, "output_subdir_clip")

        # split at cuts
        self.components.label(frame, 5, 0, "Split at Cuts",
                         tooltip="If enabled, detect cuts in the input video and split at those points. \
                            Otherwise will split at any point, and clips may contain cuts.")
        self.components.switch(frame, 5, 1, ui_state, "split_cuts")

        # maximum length
        self.components.label(frame, 6, 0, "Max Length (s)",
                         tooltip="Maximum length in seconds for saved clips, larger clips will be broken into multiple small clips.")
        self.components.entry(frame, 6, 1, ui_state, "clip_length", width=220)

        # Set FPS
        self.components.label(frame, 7, 0, "Set FPS",
                         tooltip="FPS to convert output videos to, set to 0 to keep original rate.")
        self.components.entry(frame, 7, 1, ui_state, "clip_fps", width=220)

        # Remove borders
        self.components.label(frame, 8, 0, "Remove Borders",
                         tooltip="Remove black borders from output clip")
        self.components.switch(frame, 8, 1, ui_state, "clip_bordercrop")

        # Crop Variation
        self.components.label(frame, 9, 0, "Crop Variation",
                         tooltip="Output clips will be randomly cropped to +- the base aspect ratio, \
                              somewhat biased towards making square videos. Set to 0 to use only base aspect.")
        self.components.entry(frame, 9, 1, ui_state, "clip_crop", width=220)

    def build_image_extract_tab(self, frame, controller, ui_state):
        # single video
        self.components.label(frame, 0, 0, "Single Video",
                         tooltip="Link to single video file to process.")
        self.components.path_entry(frame, 0, 1, ui_state, "image_single",
                               mode="file", allow_model_files=False, allow_video_files=True)
        self.components.button(frame, 0, 2, "Extract Single",
                          command=lambda: self._extract_images(False, controller))

        # time range
        self.components.label(frame, 1, 0, "  Time Range",
                         tooltip="Time range to limit selection for single video, \
                            format as hour:minute:second, minute:second, or seconds.")
        self.components.entry(frame, 1, 1, ui_state, "image_time_start", width=100, sticky="w")
        self.components.entry(frame, 1, 1, ui_state, "image_time_end", width=100, sticky="e")

        # directory of videos
        self.components.label(frame, 2, 0, "Directory",
                         tooltip="Path to directory with multiple videos to process, including in subdirectories.")
        self.components.path_entry(frame, 2, 1, ui_state, "image_list", mode="dir")
        self.components.button(frame, 2, 2, "Extract Directory",
                          command=lambda: self._extract_images(True, controller))

        # output directory
        self.components.label(frame, 3, 0, "Output",
                         tooltip="Path to folder where extracted images will be saved.")
        self.components.path_entry(frame, 3, 1, ui_state, "image_output", mode="dir")

        # output to subdirectories
        self.components.label(frame, 4, 0, "Output to\nSubdirectories",
                         tooltip="If enabled, files are saved to subfolders based on filename and input directory. \
                            Otherwise will all be saved to the top level of the output directory.")
        self.components.switch(frame, 4, 1, ui_state, "output_subdir_img")

        # image capture rate
        self.components.label(frame, 5, 0, "Images/sec",
                         tooltip="Number of images to capture per second of video. \
                            Images will be taken at semi-random frames around the specified frequency.")
        self.components.entry(frame, 5, 1, ui_state, "capture_rate", width=220)

        # blur removal
        self.components.label(frame, 6, 0, "Blur Removal",
                         tooltip="Threshold for removal of blurry images, relative to all others. \
                            For example at 0.2, the blurriest 20%% of the final selected frames will not be saved.")
        self.components.entry(frame, 6, 1, ui_state, "blur_threshold", width=220)

        # Remove borders
        self.components.label(frame, 7, 0, "Remove Borders",
                         tooltip="Remove black borders from output image")
        self.components.switch(frame, 7, 1, ui_state, "image_bordercrop")

        # Crop Variation
        self.components.label(frame, 8, 0, "Crop Variation",
                         tooltip="Output images will be randomly cropped to +- the base aspect ratio, \
                            somewhat biased towards making square images. Set to 0 to use only base sapect.")
        self.components.entry(frame, 8, 1, ui_state, "image_crop", width=220)

    def build_video_download_tab(self, frame, controller, ui_state):
        # link
        self.components.label(frame, 0, 0, "Single Link",
                         tooltip="Link to video/playlist to download. Uses yt-dlp, supports youtube, twitch, instagram, and many other sites.")
        self.components.entry(frame, 0, 1, ui_state, "download_link", width=220)
        self.components.button(frame, 0, 2, "Download Link",
                          command=lambda: self._download(False, controller))

        # link list
        self.components.label(frame, 1, 0, "Link List",
                         tooltip="Path to txt file with list of links separated by newlines.")
        self.components.path_entry(frame, 1, 1, ui_state, "download_list",
                               mode="file", allow_model_files=False)
        self.components.button(frame, 1, 2, "Download List",
                          command=lambda: self._download(True, controller))

        # output directory
        self.components.label(frame, 2, 0, "Output",
                         tooltip="Path to folder where downloaded videos will be saved.")
        self.components.path_entry(frame, 2, 1, ui_state, "download_output", mode="dir")

        # additional args
        self.components.label(frame, 3, 0, "Additional Args",
                         tooltip="Any additional arguments to pass to yt-dlp, for example '--restrict-filenames --force-overwrite'. \
                            Default args will hide most terminal outputs.")
        self._create_textbox(frame, 3, 1, 220, 90, ui_state, "download_args")
        self.components.button(frame, 3, 2, "yt-dlp info",
                          command=lambda: webbrowser.open("https://github.com/yt-dlp/yt-dlp?tab=readme-ov-file#usage-and-options", new=0, autoraise=False))

    @abstractmethod
    def _create_textbox(self, master, row, col, width, height, ui_state, var_name):
        pass

    @abstractmethod
    def update_status(self, status_text: str):
        pass

    @abstractmethod
    def clear_status(self):
        pass

    @abstractmethod
    def update_preview(self, preview_image, label_text: str):
        pass

    def _extract_clips(self, batch_mode: bool, controller):
        controller.extract_clips_button(batch_mode)

    def _extract_images(self, batch_mode: bool, controller):
        controller.extract_images_button(batch_mode)

    def _download(self, batch_mode: bool, controller):
        controller.download_button(batch_mode)
