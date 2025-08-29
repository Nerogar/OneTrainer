import concurrent.futures
import math
import os
import pathlib
import random
import shlex
import subprocess
import threading
import webbrowser
from tkinter import filedialog

from modules.util.path_util import SUPPORTED_VIDEO_EXTENSIONS
from modules.util.ui import components

import customtkinter as ctk
import cv2
import scenedetect


class VideoToolUI(ctk.CTkToplevel):
    def __init__(
            self,
            parent,
            *args, **kwargs,
    ):
        ctk.CTkToplevel.__init__(self, parent, *args, **kwargs)

        self.title("Video Tools")
        self.geometry("600x600")
        self.resizable(True, True)
        self.wait_visibility()
        self.focus_set()

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        tabview = ctk.CTkTabview(self)
        tabview.grid(row=0, column=0, sticky="nsew")

        self.clip_extract_tab = self.__clip_extract_tab(tabview.add("extract clips"))
        self.image_extract_tab = self.__image_extract_tab(tabview.add("extract images"))
        self.video_download_tab = self.__video_download_tab(tabview.add("download"))

    def __clip_extract_tab(self, master):
        frame = ctk.CTkScrollableFrame(master, fg_color="transparent")
        frame.grid_columnconfigure(0, weight=0, minsize=120)
        frame.grid_columnconfigure(1, weight=0, minsize=200)
        frame.grid_columnconfigure(2, weight=0)
        frame.grid_columnconfigure(3, weight=1)

        # single video
        components.label(frame, 0, 0, "Single Video",
                         tooltip="Link to single video file to process.")
        self.clip_single_entry = ctk.CTkEntry(frame, width=190)
        self.clip_single_entry.grid(row=0, column=1, sticky="w", padx=5, pady=5)
        self.clip_single_button = ctk.CTkButton(frame, width=30, text="...",
                                                command=lambda: self.__browse_for_file(self.clip_single_entry,
                                                    [("Video files", " ".join(f"*{e}" for e in SUPPORTED_VIDEO_EXTENSIONS))]
                                        ))
        self.clip_single_button.grid(row=0, column=1, sticky="e", padx=5, pady=5)
        components.button(frame, 0, 2, "Extract Single",
                          command=lambda: self.__extract_clips_button(False))

        # time range
        components.label(frame, 1, 0, "  Time Range",
                         tooltip="Time range to limit selection for single video, \
                            format as hour:minute:second, minute:second, or seconds.")
        self.clip_time_start_entry = ctk.CTkEntry(frame, width=100)
        self.clip_time_start_entry.grid(row=1, column=1, sticky="w", padx=5, pady=5)
        self.clip_time_start_entry.insert(0, "00:00:00")
        self.clip_time_end_entry = ctk.CTkEntry(frame, width=100)
        self.clip_time_end_entry.grid(row=1, column=1, sticky="e", padx=5, pady=5)
        self.clip_time_end_entry.insert(0, "99:99:99")

        # directory of videos
        components.label(frame, 2, 0, "Directory",
                         tooltip="Path to directory with multiple videos to process, including in subdirectories.")
        self.clip_list_entry = ctk.CTkEntry(frame, width=190)
        self.clip_list_entry.grid(row=2, column=1, sticky="w", padx=5, pady=5)
        self.clip_list_button = ctk.CTkButton(frame, width=30, text="...",
                                              command=lambda: self.__browse_for_dir(self.clip_list_entry))
        self.clip_list_button.grid(row=2, column=1, sticky="e", padx=5, pady=5)
        components.button(frame, 2, 2, "Extract Directory",
                          command=lambda: self.__extract_clips_button(True))

        # output directory
        components.label(frame, 3, 0, "Output",
                         tooltip="Path to folder where extracted clips will be saved.")
        self.clip_output_entry = ctk.CTkEntry(frame, width=190)
        self.clip_output_entry.grid(row=3, column=1, sticky="w", padx=5, pady=5)
        self.clip_output_button = ctk.CTkButton(frame, width=30, text="...",
                                                command=lambda: self.__browse_for_dir(self.clip_output_entry))
        self.clip_output_button.grid(row=3, column=1, sticky="e", padx=5, pady=5)

        # output to subdirectories
        self.output_subdir_clip = ctk.BooleanVar(self, False)
        components.label(frame, 4, 0, "Output to\nSubdirectories",
                         tooltip="If enabled, files are saved to subfolders based on filename and input directory. \
                            Otherwise will all be saved to the top level of the output directory.")
        self.output_subdir_clip_entry = ctk.CTkSwitch(frame, variable=self.output_subdir_clip, text="")
        self.output_subdir_clip_entry.grid(row=4, column=1, sticky="w", padx=5, pady=5)

        # split at cuts
        self.split_at_cuts = ctk.BooleanVar(self, False)
        components.label(frame, 5, 0, "Split at Cuts",
                         tooltip="If enabled, detect cuts in the input video and split at those points. \
                            Otherwise will split at any point, and clips may contain cuts.")
        self.split_cuts_entry = ctk.CTkSwitch(frame, variable=self.split_at_cuts, text="")
        self.split_cuts_entry.grid(row=5, column=1, sticky="w", padx=5, pady=5)

        # maximum length
        components.label(frame, 6, 0, "Max Length (s)",
                         tooltip="Maximum length in seconds for saved clips, larger clips will be broken into multiple small clips.")
        self.clip_length_entry = ctk.CTkEntry(frame, width=220)
        self.clip_length_entry.grid(row=6, column=1, sticky="w", padx=5, pady=5)
        self.clip_length_entry.insert(0, "3")

        # Set FPS
        components.label(frame, 7, 0, "Set FPS",
                         tooltip="FPS to convert output videos to, set to 0 to keep original rate.")
        self.clip_fps_entry = ctk.CTkEntry(frame, width=220)
        self.clip_fps_entry.grid(row=7, column=1, sticky="w", padx=5, pady=5)
        self.clip_fps_entry.insert(0, "24")

        # Remove borders
        self.clip_bordercrop = ctk.BooleanVar(self, False)
        components.label(frame, 8, 0, "Remove Borders",
                         tooltip="Remove black borders from output clip")
        self.clip_bordercrop_entry = ctk.CTkSwitch(frame, variable=self.clip_bordercrop, text="")
        self.clip_bordercrop_entry.grid(row=8, column=1, sticky="w", padx=5, pady=5)

        # Crop Variation
        components.label(frame, 9, 0, "Crop Variation",
                         tooltip="Output clips will be randomly cropped to +- the base aspect ratio, \
                              somewhat biased towards making square videos. Set to 0 to use only base aspect.")
        self.clip_crop_entry = ctk.CTkEntry(frame, width=220)
        self.clip_crop_entry.grid(row=9, column=1, sticky="w", padx=5, pady=5)
        self.clip_crop_entry.insert(0, "0.2")

        # object filter
        # components.label(frame, 9, 0, "Object Filter",
        #                  tooltip="Detect general features using Haar-Cascade classifier, and choose how to deal with clips where it is detected")
        # components.options(frame, 9, 1, ["NONE", "FACE", "EYE", "BODY"], self.video_ui_state, "filter_object")
        # components.options(frame, 9, 2, ["INCLUDE", "EXCLUDE", "SUBFOLDER"], self.video_ui_state, "filter_behavior")

        frame.pack(fill="both", expand=1)
        return frame

    def __image_extract_tab(self, master):
        frame = ctk.CTkScrollableFrame(master, fg_color="transparent")
        frame.grid_columnconfigure(0, weight=0, minsize=120)
        frame.grid_columnconfigure(1, weight=0, minsize=200)
        frame.grid_columnconfigure(2, weight=0)
        frame.grid_columnconfigure(3, weight=1)

        # single video
        components.label(frame, 0, 0, "Single Video",
                         tooltip="Link to single video file to process.")
        self.image_single_entry = ctk.CTkEntry(frame, width=190)
        self.image_single_entry.grid(row=0, column=1, sticky="w", padx=5, pady=5)
        self.image_single_button = ctk.CTkButton(frame, width=30, text="...",
                                                command=lambda: self.__browse_for_file(self.image_single_entry,
                                                    [("Video files", " ".join(f"*{e}" for e in SUPPORTED_VIDEO_EXTENSIONS))]
                                                ))
        self.image_single_button.grid(row=0, column=1, sticky="e", padx=5, pady=5)
        components.button(frame, 0, 2, "Extract Single",
                          command=lambda: self.__extract_images_button(False))

        # time range
        components.label(frame, 1, 0, "  Time Range",
                         tooltip="Time range to limit selection for single video, \
                            format as hour:minute:second, minute:second, or seconds.")
        self.image_time_start_entry = ctk.CTkEntry(frame, width=100)
        self.image_time_start_entry.grid(row=1, column=1, sticky="w", padx=5, pady=5)
        self.image_time_start_entry.insert(0, "00:00:00")
        self.image_time_end_entry = ctk.CTkEntry(frame, width=100)
        self.image_time_end_entry.grid(row=1, column=1, sticky="e", padx=5, pady=5)
        self.image_time_end_entry.insert(0, "99:99:99")

        # directory of videos
        components.label(frame, 2, 0, "Directory",
                         tooltip="Path to directory with multiple videos to process, including in subdirectories.")
        self.image_list_entry = ctk.CTkEntry(frame, width=190)
        self.image_list_entry.grid(row=2, column=1, sticky="w", padx=5, pady=5)
        self.image_list_button = ctk.CTkButton(frame, width=30, text="...",
                                               command=lambda: self.__browse_for_dir(self.image_list_entry))
        self.image_list_button.grid(row=2, column=1, sticky="e", padx=5, pady=5)
        components.button(frame, 2, 2, "Extract Directory",
                          command=lambda: self.__extract_images_button(True))

        # output directory
        components.label(frame, 3, 0, "Output",
                         tooltip="Path to folder where extracted images will be saved.")
        self.image_output_entry = ctk.CTkEntry(frame, width=190)
        self.image_output_entry.grid(row=3, column=1, sticky="w", padx=5, pady=5)
        self.image_output_button = ctk.CTkButton(frame, width=30, text="...",
                                                 command=lambda: self.__browse_for_dir(self.image_output_entry))
        self.image_output_button.grid(row=3, column=1, sticky="e", padx=5, pady=5)

        # output to subdirectories
        self.output_subdir_img = ctk.BooleanVar(self, False)
        components.label(frame, 4, 0, "Output to\nSubdirectories",
                         tooltip="If enabled, files are saved to subfolders based on filename and input directory. \
                            Otherwise will all be saved to the top level of the output directory.")
        self.output_subdir_img_entry = ctk.CTkSwitch(frame, variable=self.output_subdir_img, text="")
        self.output_subdir_img_entry.grid(row=4, column=1, sticky="w", padx=5, pady=5)

        # image capture rate
        components.label(frame, 5, 0, "Images/sec",
                         tooltip="Number of images to capture per second of video. \
                            Images will be taken at semi-random frames around the specified frequency.")
        self.capture_rate_entry = ctk.CTkEntry(frame, width=220)
        self.capture_rate_entry.grid(row=5, column=1, sticky="w", padx=5, pady=5)
        self.capture_rate_entry.insert(0, "0.5")

        # blur removal
        components.label(frame, 6, 0, "Blur Removal",
                         tooltip="Threshold for removal of blurry images, relative to all others. \
                            For example at 0.2, the blurriest 20%% of the final selected frames will not be saved.")
        self.blur_threshold_entry = ctk.CTkEntry(frame, width=220)
        self.blur_threshold_entry.grid(row=6, column=1, sticky="w", padx=5, pady=5)
        self.blur_threshold_entry.insert(0, "0.2")

        # Remove borders
        self.image_bordercrop = ctk.BooleanVar(self, False)
        components.label(frame, 7, 0, "Remove Borders",
                         tooltip="Remove black borders from output image")
        self.image_bordercrop_entry = ctk.CTkSwitch(frame, variable=self.image_bordercrop, text="")
        self.image_bordercrop_entry.grid(row=7, column=1, sticky="w", padx=5, pady=5)

        # Crop Variation
        components.label(frame, 8, 0, "Crop Variation",
                         tooltip="Output images will be randomly cropped to +- the base aspect ratio, \
                            somewhat biased towards making square images. Set to 0 to use only base sapect.")
        self.image_crop_entry = ctk.CTkEntry(frame, width=220)
        self.image_crop_entry.grid(row=8, column=1, sticky="w", padx=5, pady=5)
        self.image_crop_entry.insert(0, "0.2")

        # # object filter
        # components.label(frame, 5, 0, "Object Filter",
        #                  tooltip="Detect general features using Haar-Cascade classifier, and choose how to deal with clips where it is detected")
        # components.options(frame, 5, 1, ["NONE", "FACE", "EYE", "BODY"], self.video_ui_state, "filter_object")
        # components.options(frame, 5, 2, ["INCLUDE", "EXCLUDE", "SUBFOLDER"], self.video_ui_state, "filter_behavior")

        frame.pack(fill="both", expand=1)
        return frame

    def __video_download_tab(self, master):
        frame = ctk.CTkScrollableFrame(master, fg_color="transparent")
        frame.grid_columnconfigure(0, weight=0, minsize=120)
        frame.grid_columnconfigure(1, weight=0, minsize=200)
        frame.grid_columnconfigure(2, weight=0)
        frame.grid_columnconfigure(3, weight=1)

        # link
        components.label(frame, 0, 0, "Single Link",
                         tooltip="Link to video/playlist to download. Uses yt-dlp, supports youtube, twitch, instagram, and many other sites.")
        self.download_link_entry = ctk.CTkEntry(frame, width=220)
        self.download_link_entry.grid(row=0, column=1, sticky="w", padx=5, pady=5)
        components.button(frame, 0, 2, "Download Link", command=lambda: self.__download_button(False))

        # link list
        components.label(frame, 1, 0, "Link List",
                         tooltip="Path to txt file with list of links separated by newlines.")
        self.download_list_entry = ctk.CTkEntry(frame, width=190)
        self.download_list_entry.grid(row=1, column=1, sticky="w", padx=5, pady=5)
        self.download_list_button = ctk.CTkButton(frame, width=30, text="...",
                                                  command=lambda: self.__browse_for_file(self.download_list_entry, [("Text file", ".txt")]))
        self.download_list_button.grid(row=1, column=1, sticky="e", padx=5, pady=5)
        components.button(frame, 1, 2, "Download List", command=lambda: self.__download_button(True))

        # output directory
        components.label(frame, 2, 0, "Output",
                         tooltip="Path to folder where downloaded videos will be saved.")
        self.download_output_entry = ctk.CTkEntry(frame, width=190)
        self.download_output_entry.grid(row=2, column=1, sticky="w", padx=5, pady=5)
        self.download_output_button = ctk.CTkButton(frame, width=30, text="...", command=lambda: self.__browse_for_dir(self.download_output_entry))
        self.download_output_button.grid(row=2, column=1, sticky="e", padx=5, pady=5)

        # additional args
        components.label(frame, 3, 0, "Additional Args",
                         tooltip="Any additional arguments to pass to yt-dlp, for example '--restrict-filenames --force-overwrite'. \
                            Default args will hide most terminal outputs.")
        self.download_args_entry = ctk.CTkTextbox(frame, width=220, height=90, border_width=2)
        self.download_args_entry.grid(row=3, column=1, rowspan=2, sticky="w", padx=5, pady=5)
        self.download_args_entry.insert(index="1.0", text="--quiet --no-warnings --progress")
        components.button(frame, 3, 2, "yt-dlp info",
                          command=lambda: webbrowser.open("https://github.com/yt-dlp/yt-dlp?tab=readme-ov-file#usage-and-options", new=0, autoraise=False))

        # current status
        # self.download_label = ctk.CTkLabel(frame, text="Status:")
        # self.download_label.grid(row=5,column=0)
        # self.download_status = ctk.CTkLabel(frame, text="-")
        # self.download_status.grid(row=5,column=1)

        frame.pack(fill="both", expand=1)
        return frame

    def __browse_for_dir(self, entry_box):
        # get the path from the user
        path = filedialog.askdirectory()
        # set the path to the entry box
        # delete entry box text
        entry_box.focus_set()
        entry_box.delete(0, ctk.END)
        entry_box.insert(0, path)
        self.focus_set()

    def __browse_for_file(self, entry_box, filetypes):
        # get the path from the user
        path = filedialog.askopenfilename(filetypes=filetypes)
        # set the path to the entry box
        # delete entry box text
        entry_box.focus_set()
        entry_box.delete(0, ctk.END)
        entry_box.insert(0, path)
        self.focus_set()

    def __get_vid_paths(self, batch_mode: bool, input_path_single: str, input_path_dir: str):
        input_videos = []
        if not batch_mode:
            path = pathlib.Path(input_path_single)
            if path.is_file():
                vid = cv2.VideoCapture(str(path))
                ok = False
                try:
                    if vid.isOpened():
                        ok, _ = vid.read()
                finally:
                    vid.release()
                if ok:
                    return [path]
                else:
                    print("Invalid video file!")
                    return []
            else:
                print("No file specified, or invalid file path!")
                return []
        else:
            input_videos = []
            if not pathlib.Path(input_path_dir).is_dir() or input_path_dir == "":
                print("Invalid input directory!")
                return []
            # Only traverse supported extensions to avoid opening every file.
            lower_exts = {e.lower() for e in SUPPORTED_VIDEO_EXTENSIONS}
            for path in pathlib.Path(input_path_dir).rglob("*"):
                if path.is_file() and path.suffix.lower() in lower_exts:
                    vid = cv2.VideoCapture(str(path))
                    ok = False
                    try:
                        if vid.isOpened():
                            ok, _ = vid.read()
                    finally:
                        vid.release()
                    if ok:
                        input_videos.append(path)
            print(f'Found {len(input_videos)} videos to process')
            return input_videos

    def __get_random_aspect(self, height : int, width : int, variation: float) -> tuple[int, int, int, int]:
        if variation == 0:
            return 0, height, 0, width

        old_aspect = height/width
        variation_scaled = old_aspect*variation
        if old_aspect > 1.2:
            new_aspect = min(4.0, max(1.0, random.triangular(old_aspect-(variation_scaled*1.5), old_aspect+(variation_scaled/2), old_aspect)))
        elif old_aspect < 0.85:
            new_aspect = max(0.25, min(1.0, random.triangular(old_aspect-(variation_scaled/2), old_aspect+(variation_scaled*1.5), old_aspect)))
        else:
            new_aspect = random.triangular(old_aspect-variation_scaled, old_aspect+variation_scaled)

        new_aspect = round(new_aspect, 2)
        if new_aspect > old_aspect:
            new_height = int(height)
            new_width = int(width*(old_aspect/new_aspect))
        elif new_aspect < old_aspect:
            new_height = int(height*(new_aspect/old_aspect))
            new_width = int(width)
        else:
            new_height = int(height)
            new_width = int(width)

        position_x = random.randint(0, width-new_width)
        position_y = random.randint(0, height-new_height)
        #print(new_aspect)
        #print(position_y, new_height, position_x, new_width)
        return position_y, new_height, position_x, new_width

    def find_main_contour(self, frame):
        frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, frame_thresh = cv2.threshold(frame_grayscale, 15, 255, cv2.THRESH_BINARY)
        frame_contours, _ = cv2.findContours(frame_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if frame_contours:
            frame_maincontour = max(frame_contours, key=cv2.contourArea)
            x1, y1, w1, h1 = cv2.boundingRect(frame_maincontour)
        else:   #fallback if no contours detected
            x1 = 0
            y1 = 0
            h1, w1, _ = frame.shape
        if not frame_contours or h1 < 10 or w1 < 10:  #if bounding box did not detect the correct area, likely due to black frame
            x1 = 0
            y1 = 0
            h1, w1, _ = frame.shape
        return x1, y1, w1, h1

    def __extract_clips_button(self, batch_mode: bool):
        t = threading.Thread(target = self.__extract_clips_multi, args = [batch_mode])
        t.daemon = True
        t.start()

    def __extract_clips_multi(self, batch_mode: bool):
        if not pathlib.Path(self.clip_output_entry.get()).is_dir() or self.clip_output_entry.get() == "":
            print("Invalid output directory!")
            return

        # validate numeric inputs
        try:
            max_length = float(self.clip_length_entry.get())
            crop_variation = float(self.clip_crop_entry.get())
            target_fps = int(self.clip_fps_entry.get())
        except ValueError:
            print("Invalid numeric input for Max Length, Crop Variation, or FPS.")
            return
        if max_length <= 0.25:
            print("Max Length of clips must be > 0.25 seconds.")
            return
        if target_fps < 0:
            print("Target FPS must be a positive integer (or 0 to skip fps re-encoding).")
            return
        if not (0.0 <= crop_variation < 1.0):
            print("Crop Variation must be between 0.0 and 1.0.")
            return

        input_videos = self.__get_vid_paths(batch_mode, self.clip_single_entry.get(), self.clip_list_entry.get())
        if len(input_videos) == 0:  # exit if no paths found
            return

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            for video_path in input_videos:
                if self.output_subdir_clip_entry.get() and batch_mode:
                    output_directory = os.path.join(self.clip_output_entry.get(),
                                                    os.path.splitext(os.path.relpath(video_path, self.clip_list_entry.get()))[0])
                elif self.output_subdir_clip_entry.get() and not batch_mode:
                    output_directory = os.path.join(self.clip_output_entry.get(),
                                                    os.path.splitext(os.path.basename(video_path))[0])
                else:
                    output_directory = self.clip_output_entry.get()

                if batch_mode:
                    executor.submit(self.__extract_clips,
                                    str(video_path), "00:00:00", "99:99:99", max_length, self.split_at_cuts.get(),
                                    self.clip_bordercrop_entry.get(), crop_variation, target_fps, output_directory)
                else:
                    executor.submit(self.__extract_clips,
                                    str(video_path), str(self.clip_time_start_entry.get()), str(self.clip_time_end_entry.get()), max_length, self.split_at_cuts.get(),
                                    self.clip_bordercrop_entry.get(), crop_variation, target_fps, output_directory)

        if batch_mode:
            print(f'Clip extraction from all videos in {self.clip_list_entry.get()} complete')
        else:
            print(f'Clip extraction from {self.clip_single_entry.get()} complete')

    def __extract_clips(self, video_path: str, timestamp_min: str, timestamp_max: str, max_length: float,
                        split_at_cuts: bool, remove_borders : bool, crop_variation: float, target_fps: int, output_dir: str):
        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS) or 0.0
        if fps <= 0:
            print(f'Warning: Could not read FPS for "{os.path.basename(video_path)}". Falling back to 30 FPS.') # fallback to some sane FPS value
            fps = 30.0
        max_length_frames = int(max_length * fps)   #convert max length from seconds to frames
        min_length_frames = max(int(0.25*fps), 1)   #minimum clip length of 1/4 second or 1 frame
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        timestamp_max_frame = int(sum(int(x) * 60 ** i for i, x in enumerate(reversed(timestamp_max.split(':')))) * fps)
        timestamp_max_frame = min(timestamp_max_frame, max(total_frames - 1, 0))
        timestamp_min_frame = int(sum(int(x) * 60 ** i for i, x in enumerate(reversed(timestamp_min.split(':')))) * fps)
        timestamp_min_frame = min(timestamp_min_frame, timestamp_max_frame)

        if split_at_cuts:
            #use scenedetect to find cuts, based on start/end frame number
            timecode_list = scenedetect.detect(
                str(video_path),
                scenedetect.AdaptiveDetector(),
                start_time=int(timestamp_min_frame),
                end_time=int(timestamp_max_frame))
            scene_list = [(x[0].get_frames(), x[1].get_frames()) for x in timecode_list]
            if len(scene_list) == 0:
                scene_list = [(timestamp_min_frame, timestamp_max_frame)]    # use start/end frames if no scenes detected
        else:
            scene_list = [(timestamp_min_frame, timestamp_max_frame)]  # default if not using cuts, start and end of time range

        scene_list_split = []
        for scene in scene_list:
            length = scene[1]-scene[0]
            if length > max_length_frames:  #check for any scenes longer than max length
                n = math.ceil(length/max_length_frames) #divide into n new scenes
                new_length = int(length/n)
                new_splits = range(scene[0], scene[1]+min_length_frames, new_length)   #divide clip into closest chunks to max_length
                for i, _n in enumerate(new_splits[:-1]):
                    if new_splits[i+1] - new_splits[i] > min_length_frames:
                        scene_list_split += [(new_splits[i], new_splits[i+1])]
            else:
                if length > (min_length_frames+2):
                    scene_list_split += [(scene[0]+1, scene[1]-1)]      #trim first and last frame from detected scenes to avoid transition artifacts

        print(f'Video "{os.path.basename(video_path)}" being split into {len(scene_list_split)} clips in {output_dir}...')

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            for scene in scene_list_split:
                executor.submit(self.__save_clip, scene, video_path, target_fps, remove_borders, crop_variation, output_dir)

        video.release()

    def __save_clip(self, scene : tuple[int, int], video_path : str, target_fps : int, remove_borders : bool, crop_variation : float, output_dir : str):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        basename, ext = os.path.splitext(os.path.basename(video_path))
        video = cv2.VideoCapture(str(video_path))
        fps = video.get(cv2.CAP_PROP_FPS) or 0.0
        if fps <= 0:
            print(f'Warning: Could not read FPS for "{os.path.basename(video_path)}". Falling back to 30 FPS.')
            fps = 30.0
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_name = f'{output_dir}{os.sep}{basename}_{scene[0]}-{scene[1]}'
        output_ext = ".mp4"

        video.set(cv2.CAP_PROP_POS_FRAMES, (scene[1] + scene[0])//2)
        frame_number = int(video.get(cv2.CAP_PROP_POS_FRAMES))
        success, frame = video.read()
        if not success or frame is None:
            print(f'Failed to read frame from "{os.path.basename(video_path)}" at {int(frame_number)}. Skipping clip.')
            video.release()
            return

        #crop out borders of frame - blends five random frames from the scene to get "average" image
        #helps prevent incorrect cropping when sampled frame may be all black or otherwise detect incorrect border
        if remove_borders:
            frame_blend = frame
            for i in range(5):  # blend 5 random frames to get average
                random_frame = random.randint(scene[0], scene[1])
                video.set(cv2.CAP_PROP_POS_FRAMES, random_frame)
                success, frame = video.read()
                if not success or frame is None:
                    continue
                a = 1/(i+1)
                b = 1-a
                frame_blend = cv2.addWeighted(frame, a, frame_blend, b, 0)
            x1, y1, w1, h1 = self.find_main_contour(frame_blend)
        else:
            x1 = 0
            y1 = 0
            h1, w1, _ = frame.shape

        y2, h2, x2, w2 = self.__get_random_aspect(h1, w1, crop_variation)
        writer = cv2.VideoWriter(output_name+output_ext, fourcc, fps, (w2, h2))
        video.set(cv2.CAP_PROP_POS_FRAMES, scene[0])
        frame_number = int(video.get(cv2.CAP_PROP_POS_FRAMES))
        success, frame = video.read()

        while success and (frame_number < scene[1]):    # loop through frames within each scene
            frame_trimmed = frame[y1:y1+h1, x1:x1+w1]   # cut out black borders if applicable
            writer.write(frame_trimmed[y2:y2+h2, x2:x2+w2]) # save frame with random crop variation if applicable
            success, frame = video.read()
            frame_number += 1
        writer.release()
        video.release()

        if target_fps > 0: # use ffmpeg to change to set framerate - saves copy and deletes original
            if int(round(fps)) == target_fps:
                # Already at desired fps; skip re-encode.
                return
            cmd = [
                "ffmpeg", "-y",
                "-i", f"{output_name}{output_ext}",
                "-filter:v", f"fps={target_fps}",
                "-an",
                f"{output_name}_{target_fps}fps{output_ext}",
            ]
            proc = subprocess.run(cmd, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            if proc.returncode == 0:
                try:
                    os.remove(output_name + output_ext)
                except OSError:
                    print(f"Failed to remove conversion placeholder {output_name + output_ext}, remove manually or check folder permissions.")

    def __extract_images_button(self, batch_mode : bool):
        t = threading.Thread(target = self.__extract_images_multi, args = [batch_mode])
        t.daemon = True
        t.start()

    def __extract_images_multi(self, batch_mode : bool):
        if not pathlib.Path(self.image_output_entry.get()).is_dir() or self.image_output_entry.get() == "":
            print("Invalid output directory!")
            return

        # validate numeric inputs
        try:
            capture_rate = float(self.capture_rate_entry.get())
            blur_threshold = float(self.blur_threshold_entry.get())
            crop_variation = float(self.image_crop_entry.get())
        except ValueError:
            print("Invalid numeric input for Images/sec, Blur Removal, or Crop Variation.")
            return
        if capture_rate <= 0:
            print("Images/sec must be > 0.")
            return
        if not (0.0 <= blur_threshold < 1.0):
            print("Blur Removal must be between 0.0 and 1.0.")
            return
        if not (0.0 <= crop_variation < 1.0):
            print("Crop Variation must be between 0.0 and 1.0.")
            return

        input_videos = self.__get_vid_paths(batch_mode, self.image_single_entry.get(), self.image_list_entry.get())
        if len(input_videos) == 0:  #exit if no paths found
            return

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            for video_path in input_videos:
                if self.output_subdir_img_entry.get() and batch_mode:
                    output_directory = os.path.join(self.image_output_entry.get(),
                                                    os.path.splitext(os.path.relpath(video_path, self.image_list_entry.get()))[0])
                elif self.output_subdir_img_entry.get() and not batch_mode:
                    output_directory = os.path.join(self.image_output_entry.get(),
                                                    os.path.splitext(os.path.basename(video_path))[0])
                else:
                    output_directory = self.image_output_entry.get()

                if batch_mode:
                    executor.submit(self.__save_frames,
                                    str(video_path), "00:00:00", "99:99:99", capture_rate,
                                    blur_threshold, self.image_bordercrop.get(), crop_variation, output_directory)
                else:
                    executor.submit(self.__save_frames,
                                    str(video_path), str(self.image_time_start_entry.get()), str(self.image_time_end_entry.get()), capture_rate,
                                    blur_threshold, self.image_bordercrop.get(), crop_variation, output_directory)
        if batch_mode:
            print(f'Image extraction from all videos in {self.image_list_entry.get()} complete')
        else:
            print(f'Image extraction from {self.image_single_entry.get()} complete')

    def __save_frames(self, video_path: str, timestamp_min: str, timestamp_max: str, capture_rate: float,
                      blur_threshold: float, remove_borders : bool, crop_variation: float, output_dir: str):
        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS) or 0.0
        if fps <= 0:
            print(f'Warning: Could not read FPS for "{os.path.basename(video_path)}". Falling back to 30 FPS.')
            fps = 30.0
        if capture_rate <= 0:
            print("Images/sec must be > 0.")
            video.release()
            return
        image_rate = max(int(fps / capture_rate), 1)   # frames between captures (min 1)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        timestamp_max_frame = int(sum(int(x) * 60 ** i for i, x in enumerate(reversed(timestamp_max.split(':')))) * fps)
        timestamp_max_frame = min(timestamp_max_frame, max(total_frames - 1, 0))
        timestamp_min_frame = int(sum(int(x) * 60 ** i for i, x in enumerate(reversed(timestamp_min.split(':')))) * fps)
        timestamp_min_frame = min(timestamp_min_frame, timestamp_max_frame)
        frame_range = range(timestamp_min_frame, timestamp_max_frame, image_rate)
        frame_list = []

        for n in frame_range:
            frame = abs(int(random.triangular(n-(image_rate/2), n+(image_rate/2))))     #random triangular distribution around center
            frame = max(0, min(frame, max(total_frames - 1, 0)))
            frame_list.append(frame)

        print(f'Video "{os.path.basename(video_path)}" will be split into {len(frame_list)} images in {output_dir}...')

        output_list = []
        for f in frame_list:
            video.set(cv2.CAP_PROP_POS_FRAMES, f)
            success, frame = video.read()
            if success and frame is not None:
                frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame_sharpness = cv2.Laplacian(frame_grayscale, cv2.CV_64F).var()  #get sharpness of greyscale pic
                output_list.append((f, frame_sharpness))

        if not output_list:
            print(f'No frames extracted from {os.path.basename(video_path)} in the selected range.')
            video.release()
            return

        output_list_sorted = sorted(output_list, key=lambda x: x[1])
        cutoff = int(blur_threshold*len(output_list_sorted))     #calculate cutoff as portion of total frames
        output_list_cut = output_list_sorted[cutoff:]            # keep all frames above cutoff
        print(f'{cutoff} blurriest images have been dropped from {os.path.basename(video_path)}')

        basename, ext = os.path.splitext(os.path.basename(video_path))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for f in output_list_cut:
            filename = f'{output_dir}{os.sep}{basename}_{f[0]}.jpg'
            video.set(cv2.CAP_PROP_POS_FRAMES, f[0])
            success, frame = video.read()

            #crop out borders of frame
            if remove_borders and success and frame is not None:
                x1, y1, w1, h1 = self.find_main_contour(frame)
                frame_cropped = frame[y1:y1+h1, x1:x1+w1]
            else:
                frame_cropped = frame if success and frame is not None else None
                if frame_cropped is not None:
                    x1 = 0
                    y1 = 0
                    h1, w1, _ = frame_cropped.shape

            y2, h2, x2, w2 = self.__get_random_aspect(h1, w1, crop_variation)
            #print(y1, h1, x1, w1, ":", y2, h2, x2, w2)

            if success and frame is not None:
                cv2.imwrite(filename, frame_cropped[y2:y2+h2, x2:x2+w2])    #save images
        video.release()

    def __download_button(self, batch_mode: bool):
        t = threading.Thread(target = self.__download_multi, args = [batch_mode])
        t.daemon = True
        t.start()

    def __download_multi(self, batch_mode: bool):
        if not pathlib.Path(self.download_output_entry.get()).is_dir() or self.download_output_entry.get() == "":
            print("Invalid output directory!")
            return

        if not batch_mode:
            ydl_urls = [self.download_link_entry.get()]
        elif batch_mode:
            ydl_path = pathlib.Path(self.download_list_entry.get())
            if ydl_path.is_file() and ydl_path.suffix.lower() == ".txt":
                with open(ydl_path) as file:
                    ydl_urls = file.readlines()
            else:
                print("Invalid link list!")
                return

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            for url in ydl_urls:
                executor.submit(self.__download_video,
                                url.strip(), self.download_output_entry.get(), self.download_args_entry.get("0.0", ctk.END))

        print(f'Completed {len(ydl_urls)} downloads.')

    def __download_video(self, url: str, output_dir: str, output_args: str):
        url = (url or "").strip()
        if not url:
            print("Empty URL, skipping download.")
            return

        additional_args = shlex.split(output_args.strip()) if output_args and output_args.strip() else [] # Respect quotes and split into list
        cmd = ["yt-dlp", "-o", "%(title)s.%(ext)s", "-P", output_dir] + additional_args + [url]

        print(f'Downloading {url}...')
        subprocess.run(cmd)
        print(f'Download {url} done!')
