import subprocess
from tkinter import filedialog

from modules.util.ui import components
from modules.util.ui.UIState import UIState

import customtkinter as ctk


class VideoToolUI(ctk.CTkToplevel):
    def __init__(
            self,
            parent,
            video_ui_state: UIState,
            *args, **kwargs,
    ):
        ctk.CTkToplevel.__init__(self, parent, *args, **kwargs)

        self.video_ui_state = video_ui_state

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

        #components.button(self, 1, 0, "ok", self.__ok)

    def __clip_extract_tab(self, master):
        frame = ctk.CTkScrollableFrame(master, fg_color="transparent")
        frame.grid_columnconfigure(0, weight=0)
        frame.grid_columnconfigure(1, weight=0)
        frame.grid_columnconfigure(2, weight=0)
        frame.grid_columnconfigure(3, weight=1)

        # input directory
        components.label(frame, 0, 0, "Input",
                         tooltip="Path to folder or single video file to process")
        components.dir_entry(frame, 0, 1, self.video_ui_state, "input")

        # include subdirectories
        components.label(frame, 1, 0, "Include Subdirectories",
                         tooltip="Includes videos from subdirectories")
        components.switch(frame, 1, 1, self.video_ui_state, "include_subdirectories")

        # output directory
        components.label(frame, 2, 0, "Output",
                         tooltip="Path to folder where clips will be saved")
        components.dir_entry(frame, 2, 1, self.video_ui_state, "video_output")

        # split at cuts
        components.label(frame, 3, 0, "Split at Cuts",
                         tooltip="If enabled, detect cuts in the input video and split at those points. Otherwise will split at random.")
        components.switch(frame, 3, 1, self.video_ui_state, "cut_split_enabled")

        # maximum length
        components.label(frame, 4, 0, "Maximum Length",
                         tooltip="Maximum length in seconds for saved clips, larger clips will be broken into multiple small clips.")
        components.entry(frame, 4, 1, self.video_ui_state, "maximum_length")

        # prompt source
        components.label(frame, 5, 0, "Object Filter",
                         tooltip="Detect general features using Haar-Cascade classifier, and choose how to deal with clips where it is detected")
        components.options(frame, 5, 1, ["NONE", "FACE", "EYE", "BODY"], self.video_ui_state, "filter_object")
        components.options(frame, 5, 2, ["INCLUDE", "EXCLUDE", "SUBFOLDER"], self.video_ui_state, "filter_behavior")

        frame.pack(fill="both", expand=1)
        return frame

    def __image_extract_tab(self, master):
        frame = ctk.CTkScrollableFrame(master, fg_color="transparent")
        frame.grid_columnconfigure(0, weight=0)
        frame.grid_columnconfigure(1, weight=0)
        frame.grid_columnconfigure(2, weight=0)
        frame.grid_columnconfigure(3, weight=1)

        # input directory
        components.label(frame, 0, 0, "Input",
                         tooltip="Path to folder or single video file to process")
        components.dir_entry(frame, 0, 1, self.video_ui_state, "input")

        # include subdirectories
        components.label(frame, 1, 0, "Include Subdirectories",
                         tooltip="Includes videos from subdirectories")
        components.switch(frame, 1, 1, self.video_ui_state, "include_subdirectories")

        # output directory
        components.label(frame, 2, 0, "Output",
                         tooltip="Path to folder where images will be saved")
        components.dir_entry(frame, 2, 1, self.video_ui_state, "image_output")

        # image capture rate
        components.label(frame, 3, 0, "Images/sec",
                         tooltip="Number of images to capture per second of video. Images will be taken at semi-random frames around the specified frequency")
        components.entry(frame, 3, 1, self.video_ui_state, "image_rate")

        # maximum length
        components.label(frame, 4, 0, "Blur Removal",
                         tooltip="Threshold for removal of blurry images, relative to all others. For example at 0.4, the blurriest 40%% of the final images captured will be deleted.")
        components.entry(frame, 4, 1, self.video_ui_state, "blur_threshold")

        # prompt source
        components.label(frame, 5, 0, "Object Filter",
                         tooltip="Detect general features using Haar-Cascade classifier, and choose how to deal with clips where it is detected")
        components.options(frame, 5, 1, ["NONE", "FACE", "EYE", "BODY"], self.video_ui_state, "filter_object")
        components.options(frame, 5, 2, ["INCLUDE", "EXCLUDE", "SUBFOLDER"], self.video_ui_state, "filter_behavior")

        frame.pack(fill="both", expand=1)
        return frame

    def __video_download_tab(self, master):
        frame = ctk.CTkScrollableFrame(master, fg_color="transparent")
        frame.grid_columnconfigure(0, weight=0)
        frame.grid_columnconfigure(1, minsize=250, weight=0)
        frame.grid_columnconfigure(2, weight=1)

        # link
        components.label(frame, 0, 0, "Single Link",
                         tooltip="Link to video/playlist to download. Uses yt-dlp, supports youtube, twitch, instagram, and many other sites.")
        self.download_link_entry = ctk.CTkEntry(frame, width=220)
        self.download_link_entry.grid(row=0, column=1, sticky="w", padx=5, pady=5)

        # link list
        components.label(frame, 1, 0, "Link List",
                         tooltip="Path to txt file with list of links separated by newlines.")
        self.download_list_entry = ctk.CTkEntry(frame, width=190)
        self.download_list_entry.grid(row=1, column=1, sticky="w", padx=5, pady=5)
        self.download_list_button = ctk.CTkButton(frame, width=30, text="...", command=lambda: self.__browse_for_file(self.download_list_entry))
        self.download_list_button.grid(row=1, column=1, sticky="e", padx=5, pady=5)

        # output directory
        components.label(frame, 2, 0, "Output",
                         tooltip="Path to folder where downloaded videos will be saved.")
        self.download_output_entry = ctk.CTkEntry(frame, width=190)
        self.download_output_entry.grid(row=2, column=1, sticky="w", padx=5, pady=5)
        self.download_output_button = ctk.CTkButton(frame, width=30, text="...", command=lambda: self.__browse_for_dir(self.download_output_entry))
        self.download_output_button.grid(row=2, column=1, sticky="e", padx=5, pady=5)

        # additional args
        components.label(frame, 3, 0, "Additional Args",
                         tooltip="Any additional arguments to pass to yt-dlp, for example '--restrict-filenames --force-overwrite'")
        self.download_args_entry = ctk.CTkTextbox(frame, width=220, height=90, border_width=2)
        self.download_args_entry.grid(row=3, column=1, rowspan=2, sticky="w", padx=5, pady=5)

        # download link
        components.button(frame, 0, 3, "Download Link", command=lambda: self.__download_video(False))

        # download lst
        components.button(frame, 1, 3, "Download List", command=lambda: self.__download_video(True))

        # current status
        self.download_label = ctk.CTkLabel(frame, text="Status:")
        self.download_label.grid(row=5,column=0)
        self.download_status = ctk.CTkLabel(frame, text="-")
        self.download_status.grid(row=5,column=1)

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

    def __browse_for_file(self, entry_box):
        # get the path from the user
        path = filedialog.askopenfilename(filetypes=[("Text file", ".txt")])
        # set the path to the entry box
        # delete entry box text
        entry_box.focus_set()
        entry_box.delete(0, filedialog.END)
        entry_box.insert(0, path)
        self.focus_set()

    def __download_video(self, batch_mode : bool):
        if not batch_mode:
            ydl_urls = [self.download_link_entry.get()]
        elif batch_mode:
            with open(self.download_list_entry.get()) as file:
                ydl_urls = file.readlines()
        ydl_output = "-o %(title)s.%(ext)s"
        ydl_path = '-P ' + self.download_output_entry.get()
        ydl_args = self.download_args_entry.get("0.0", ctk.END).split()

        error_count = 0
        for index, url in enumerate(ydl_urls, start=1):
            try:
                self.download_status.configure(text=f"Download {index}/{len(ydl_urls)} running...")
                subprocess.run(["yt-dlp", ydl_output, ydl_path, url] + ydl_args)
                self.download_status.configure(text=f"Download {index}/{len(ydl_urls)} complete!")
            except subprocess.CalledProcessError as e:  # noqa: PERF203
                self.download_status.configure(text=f"Download {index}/{len(ydl_urls)} error: {e}")
                error_count += 1
                continue

        self.download_status.configure(text=f"{len(ydl_urls)} downloads complete, {error_count} errors")
