import concurrent.futures
import math
import os
import pathlib
import random
import subprocess
import threading
import webbrowser
from tkinter import filedialog

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
                                                command=lambda: self.__browse_for_file(self.clip_single_entry, [("Video file",".*")]))
        self.clip_single_button.grid(row=0, column=1, sticky="e", padx=5, pady=5)
        components.button(frame, 0, 2, "Extract Single",
                          command=lambda: self.__extract_clips_button(False))

        # directory of videos
        components.label(frame, 1, 0, "Directory",
                         tooltip="Path to directory with multiple videos to process, including in subdirectories.")
        self.clip_list_entry = ctk.CTkEntry(frame, width=190)
        self.clip_list_entry.grid(row=1, column=1, sticky="w", padx=5, pady=5)
        self.clip_list_button = ctk.CTkButton(frame, width=30, text="...",
                                              command=lambda: self.__browse_for_dir(self.clip_list_entry))
        self.clip_list_button.grid(row=1, column=1, sticky="e", padx=5, pady=5)
        components.button(frame, 1, 2, "Extract Directory",
                          command=lambda: self.__extract_clips_button(True))

        # output directory
        components.label(frame, 2, 0, "Output",
                         tooltip="Path to folder where extracted clips will be saved.")
        self.clip_output_entry = ctk.CTkEntry(frame, width=190)
        self.clip_output_entry.grid(row=2, column=1, sticky="w", padx=5, pady=5)
        self.clip_output_button = ctk.CTkButton(frame, width=30, text="...",
                                                command=lambda: self.__browse_for_dir(self.clip_output_entry))
        self.clip_output_button.grid(row=2, column=1, sticky="e", padx=5, pady=5)

        # output to subdirectories
        self.output_subdir_clip = ctk.BooleanVar(self, False)
        components.label(frame, 3, 0, "Output to\nSubdirectories",
                         tooltip="If enabled, files are saved to subfolders based on filename and input directory. \
                            Otherwise will all be saved to the top level of the output directory.")
        self.output_subdir_clip_entry = ctk.CTkSwitch(frame, variable=self.output_subdir_clip, text="")
        self.output_subdir_clip_entry.grid(row=3, column=1, sticky="w", padx=5, pady=5)

        # split at cuts
        self.split_at_cuts = ctk.BooleanVar(self, False)
        components.label(frame, 4, 0, "Split at Cuts",
                         tooltip="If enabled, detect cuts in the input video and split at those points. \
                            Otherwise will split at any point, and clips may contain cuts.")
        self.split_cuts_entry = ctk.CTkSwitch(frame, variable=self.split_at_cuts, text="")
        self.split_cuts_entry.grid(row=4, column=1, sticky="w", padx=5, pady=5)

        # maximum length
        components.label(frame, 5, 0, "Max Length (s)",
                         tooltip="Maximum length in seconds for saved clips, larger clips will be broken into multiple small clips.")
        self.clip_length_entry = ctk.CTkEntry(frame, width=220)
        self.clip_length_entry.grid(row=5, column=1, sticky="w", padx=5, pady=5)
        self.clip_length_entry.insert(0, "3")

        # object filter
        # components.label(frame, 5, 0, "Object Filter",
        #                  tooltip="Detect general features using Haar-Cascade classifier, and choose how to deal with clips where it is detected")
        # components.options(frame, 5, 1, ["NONE", "FACE", "EYE", "BODY"], self.video_ui_state, "filter_object")
        # components.options(frame, 5, 2, ["INCLUDE", "EXCLUDE", "SUBFOLDER"], self.video_ui_state, "filter_behavior")

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
                                                 command=lambda: self.__browse_for_file(self.image_single_entry, [("Video file",".*")]))
        self.image_single_button.grid(row=0, column=1, sticky="e", padx=5, pady=5)
        components.button(frame, 0, 2, "Extract Single",
                          command=lambda: self.__extract_images_button(False))

        # directory of videos
        components.label(frame, 1, 0, "Directory",
                         tooltip="Path to directory with multiple videos to process, including in subdirectories.")
        self.image_list_entry = ctk.CTkEntry(frame, width=190)
        self.image_list_entry.grid(row=1, column=1, sticky="w", padx=5, pady=5)
        self.image_list_button = ctk.CTkButton(frame, width=30, text="...",
                                               command=lambda: self.__browse_for_dir(self.image_list_entry))
        self.image_list_button.grid(row=1, column=1, sticky="e", padx=5, pady=5)
        components.button(frame, 1, 2, "Extract Directory",
                          command=lambda: self.__extract_images_button(True))

        # output directory
        components.label(frame, 2, 0, "Output",
                         tooltip="Path to folder where extracted images will be saved.")
        self.image_output_entry = ctk.CTkEntry(frame, width=190)
        self.image_output_entry.grid(row=2, column=1, sticky="w", padx=5, pady=5)
        self.image_output_button = ctk.CTkButton(frame, width=30, text="...",
                                                 command=lambda: self.__browse_for_dir(self.image_output_entry))
        self.image_output_button.grid(row=2, column=1, sticky="e", padx=5, pady=5)

        # output to subdirectories
        self.output_subdir_img = ctk.BooleanVar(self, False)
        components.label(frame, 3, 0, "Output to\nSubdirectories",
                         tooltip="If enabled, files are saved to subfolders based on filename and input directory. \
                            Otherwise will all be saved to the top level of the output directory.")
        self.output_subdir_img_entry = ctk.CTkSwitch(frame, variable=self.output_subdir_img, text="")
        self.output_subdir_img_entry.grid(row=3, column=1, sticky="w", padx=5, pady=5)

        # image capture rate
        components.label(frame, 4, 0, "Images/sec",
                         tooltip="Number of images to capture per second of video. \
                            Images will be taken at semi-random frames around the specified frequency.")
        self.capture_rate_entry = ctk.CTkEntry(frame, width=220)
        self.capture_rate_entry.grid(row=4, column=1, sticky="w", padx=5, pady=5)
        self.capture_rate_entry.insert(0, "0.5")

        # blur removal
        components.label(frame, 5, 0, "Blur Removal",
                         tooltip="Threshold for removal of blurry images, relative to all others. \
                            For example at 0.2, the blurriest 20%% of the final selected frames will not be saved.")
        self.blur_threshold_entry = ctk.CTkEntry(frame, width=220)
        self.blur_threshold_entry.grid(row=5, column=1, sticky="w", padx=5, pady=5)
        self.blur_threshold_entry.insert(0, "0.2")

        # object filter
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
        entry_box.delete(0, filedialog.END)
        entry_box.insert(0, path)
        self.focus_set()

    def __browse_for_file(self, entry_box, type):
        # get the path from the user
        path = filedialog.askopenfilename(filetypes=type)
        # set the path to the entry box
        # delete entry box text
        entry_box.focus_set()
        entry_box.delete(0, filedialog.END)
        entry_box.insert(0, path)
        self.focus_set()

    def __get_vid_paths(self, batch_mode : bool, input_path_single : str, input_path_dir : str):
        input_videos = []
        if not batch_mode:
            path = pathlib.Path(input_path_single)
            if path.is_file():
                vid = cv2.VideoCapture(path)
                if vid.isOpened() and vid.read()[0]:    #check if valid video
                    input_videos = [path]
                    vid.release()
                    return input_videos
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
            for path in pathlib.Path(input_path_dir).glob("**/*.*"):    #check directory and subdirectories
                if path.is_file():
                    vid = cv2.VideoCapture(path)
                    if vid.isOpened() and vid.read()[0]:    #check if valid video
                        input_videos += [path]
                    vid.release()
            print(f'Found {len(input_videos)} videos to process')
            return input_videos

    def __extract_clips_button(self, batch_mode : bool):
        t = threading.Thread(target = self.__extract_clips_multi, args = [batch_mode])
        t.daemon = True
        t.start()

    def __extract_clips_multi(self, batch_mode : bool):
        if not pathlib.Path(self.clip_output_entry.get()).is_dir() or self.clip_output_entry.get() == "":
            print("Invalid output directory!")
            return

        input_videos = self.__get_vid_paths(batch_mode, self.clip_single_entry.get(), self.clip_list_entry.get())
        if len(input_videos) == 0:  #exit if no paths found
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

                executor.submit(self.__extract_clips,
                                str(video_path), float(self.clip_length_entry.get()), self.split_at_cuts.get(), output_directory)

        print("Clip extraction from all videos complete")

    def __extract_clips(self, video_path : str, max_length : float, split_at_cuts : bool, output_dir : str):
        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        max_length_frames = max_length * fps    #convert max length from seconds to frames
        min_length_frames = int(0.25*fps)    #minimum clip length of 1/4 second

        if split_at_cuts:
            timecode_list = scenedetect.detect(str(video_path), scenedetect.AdaptiveDetector()) #detect scene transitions
            scene_list = [(x[0].get_frames(), x[1].get_frames()) for x in timecode_list]
            if len(scene_list) == 0:
                scene_list = [(0,int(video.get(cv2.CAP_PROP_FRAME_COUNT)))]     #use start/end frames if no scenes detected
        else:
            scene_list = [(0,int(video.get(cv2.CAP_PROP_FRAME_COUNT)))]  #default if not using cuts, start and end of entire video

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
                executor.submit(self.__save_clip, scene, video_path, output_dir)

        video.release()

    def __save_clip(self, scene : tuple[int, int], video_path : str, output_dir : str):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        basename, ext = os.path.splitext(os.path.basename(video_path))
        video = cv2.VideoCapture(video_path)
        size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = video.get(cv2.CAP_PROP_FPS)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        writer = cv2.VideoWriter(f'{output_dir}{os.sep}{basename}_{scene[0]}-{scene[1]}.avi',fourcc,fps,size)
        video.set(cv2.CAP_PROP_POS_FRAMES, scene[0])
        frame_number = video.get(cv2.CAP_PROP_POS_FRAMES)
        success, frame = video.read()
        while success and (frame_number < scene[1]):    #loop through frames within each scene
            writer.write(frame)
            success, frame = video.read()
            frame_number += 1
        writer.release()
        video.release()

    def __extract_images_button(self, batch_mode : bool):
        t = threading.Thread(target = self.__extract_images_multi, args = [batch_mode])
        t.daemon = True
        t.start()

    def __extract_images_multi(self, batch_mode : bool):
        if not pathlib.Path(self.image_output_entry.get()).is_dir() or self.image_output_entry.get() == "":
            print("Invalid output directory!")
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

                executor.submit(self.__save_frames,
                                video_path, float(self.capture_rate_entry.get()), float(self.blur_threshold_entry.get()), output_directory)

        print("Image extraction from all videos complete")

    def __save_frames(self, video_path : str, capture_rate : float, blur_threshold : float, output_dir : str):
        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        image_rate = int(fps / capture_rate)   #convert capture rate from seconds to frames
        frame_range = range(0,int(video.get(cv2.CAP_PROP_FRAME_COUNT)), image_rate)
        frame_list = []

        for n in frame_range:
            frame = abs(int(random.triangular(n-(image_rate/2), n+(image_rate/2))))     #random triangular distribution around center
            frame_list += [min(frame, int(video.get(cv2.CAP_PROP_FRAME_COUNT)))]

        print(f'Video "{os.path.basename(video_path)}" will be split into {len(frame_list)} images in {output_dir}...')

        output_list = []
        for f in frame_list:
            video.set(cv2.CAP_PROP_POS_FRAMES, f)
            success, frame = video.read()
            if success:
                frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame_sharpness = cv2.Laplacian(frame_grayscale, cv2.CV_64F).var()  #get sharpness of greyscale pic
                output_list += [(f, frame_sharpness)]

        output_list_sorted = sorted(output_list, key=lambda x: x[1])
        cutoff = int(blur_threshold*len(output_list_sorted))     #calculate cutoff as portion of total frames
        output_list_cut = output_list_sorted[cutoff:-1]     #drop the lowest sharpness values
        print(f'{cutoff} blurriest images have been dropped from {os.path.basename(video_path)}')

        basename, ext = os.path.splitext(os.path.basename(video_path))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for f in output_list_cut:
            filename = f'{output_dir}{os.sep}{basename}_{f[0]}.jpg'
            video.set(cv2.CAP_PROP_POS_FRAMES, f[0])
            success, frame = video.read()
            if success:
                cv2.imwrite(filename, frame)    #save images
        video.release()

    def __download_button(self, batch_mode : bool):
        t = threading.Thread(target = self.__download_multi, args = [batch_mode])
        t.daemon = True
        t.start()

    def __download_multi(self, batch_mode : bool):
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

    def __download_video(self, url : str, output_dir : str, output_args : str):
        ydl_filename = '-o%(title)s.%(ext)s'
        ydl_outpath = '-P ' + output_dir
        ydl_args = output_args.split()     #split args into list

        print(f'Downloading {url}...')
        subprocess.run(["yt-dlp", ydl_filename, ydl_outpath, url] + ydl_args)
        print(f'Download {url} done!')
