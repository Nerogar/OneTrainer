import concurrent.futures
import math
import os
import pathlib
import random
import shlex
import shutil
import subprocess

from modules.ui.models.SingletonConfigModel import SingletonConfigModel
from modules.util.path_util import SUPPORTED_VIDEO_EXTENSIONS

import cv2
import scenedetect


class VideoModel(SingletonConfigModel):
    def __init__(self):
        super().__init__({
            "clips": {
                "single_video": "",
                "range_start": "00:00:00",
                "range_end": "99:99:99",
                "directory": "",
                "output": "",
                "output_to_subdirectories": False,
                "split_at_cuts": False,
                "max_length": 0,
                "fps": 0,
                "remove_borders": False,
                "crop_variation": 0
            },
            "images": {
                "single_video": "",
                "range_start": "00:00:00",
                "range_end": "99:99:99",
                "directory": "",
                "output": "",
                "output_to_subdirectories": False,
                "capture_rate": 0,
                "blur_removal": 0,
                "remove_borders": False,
                "crop_variation": 0
            },
            "download": {
                "single_link": "",
                "link_list": "",
                "output": "",
                "additional_args": "--quiet --no-warnings --progress"
            }
        })

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
                    self.log("error", "Invalid video file!")
                    return []
            else:
                self.log("error", "No file specified, or invalid file path!")
                return []
        else:
            input_videos = []
            if not pathlib.Path(input_path_dir).is_dir() or input_path_dir == "":
                self.log("error", "Invalid input directory!")
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
            self.log("info", f'Found {len(input_videos)} videos to process')
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

    def __find_main_contour(self, frame):
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

    def extract_clips_multi(self, batch_mode: bool):
        cfg = self.bulk_read("clips.output", "clips.max_length", "clips.crop_variation",
                             "clips.fps", "clips.single_video", "clips.directory",
                             "clips.output_to_subdirectories", "clips.split_at_cuts",
                             "clips.remove_borders", "clips.range_start", "clips.range_end",
                             as_dict=True)


        # if not pathlib.Path(cfg["clips.output"]).is_dir() or cfg["clips.output"] == "":
        #     self.log("error", "Invalid output directory!")
        #     return
        pathlib.Path(cfg["clips.output"]).mkdir(parents=True, exist_ok=True)


        # validate numeric inputs
        try:
            max_length = float(cfg["clips.max_length"])
            crop_variation = float(cfg["clips.crop_variation"])
            target_fps = int(cfg["clips.fps"])
        except ValueError:
            self.log("error", "Invalid numeric input for Max Length, Crop Variation, or FPS.")
            return
        # if max_length <= 0.25:
        #     self.log("error", "Max Length of clips must be > 0.25 seconds.")
        #     return
        # if target_fps < 0:
        #     self.log("error", "Target FPS must be a positive integer (or 0 to skip fps re-encoding).")
        #     return
        # if not (0.0 <= crop_variation < 1.0):
        #     self.log("error", "Crop Variation must be between 0.0 and 1.0.")
        #     return

        input_videos = self.__get_vid_paths(batch_mode, cfg["clips.single_video"], cfg["clips.directory"])
        if len(input_videos) == 0:  # exit if no paths found
            return

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            for video_path in input_videos:
                if cfg["clips.output_to_subdirectories"] and batch_mode:
                    output_directory = os.path.join(cfg["clips.output"],
                                                    os.path.splitext(os.path.relpath(video_path, cfg["clips.directory"]))[0])
                elif cfg["clips.output_to_subdirectories"] and not batch_mode:
                    output_directory = os.path.join(cfg["clips.output"],
                                                    os.path.splitext(os.path.basename(video_path))[0])
                else:
                    output_directory = cfg["clips.output"]

                if batch_mode:
                    executor.submit(self.__extract_clips,
                                    str(video_path), "00:00:00", "99:99:99", max_length, cfg["clips.split_at_cuts"],
                                    cfg["clips.remove_borders"], crop_variation, target_fps, output_directory)
                else:
                    executor.submit(self.__extract_clips,
                                    str(video_path), str(cfg["clips.range_start"]), str(cfg["clips.range_end"]), max_length, cfg["clips.split_at_cuts"],
                                    cfg["clips.remove_borders"], crop_variation, target_fps, output_directory)

        if batch_mode:
            self.log("info", f'Clip extraction from all videos in {cfg["clips.directory"]} complete')
        else:
            self.log("info", f'Clip extraction from {cfg["clips.single_video"]} complete')

    def __extract_clips(self, video_path: str, timestamp_min: str, timestamp_max: str, max_length: float,
                        split_at_cuts: bool, remove_borders : bool, crop_variation: float, target_fps: int, output_dir: str):
        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS) or 0.0
        if fps <= 0:
            self.log("warning", f'Could not read FPS for "{os.path.basename(video_path)}". Falling back to 30 FPS.') # fallback to some sane FPS value
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

        self.log("info", f'Video "{os.path.basename(video_path)}" being split into {len(scene_list_split)} clips in {output_dir}...')

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
            self.log("warning", f'Could not read FPS for "{os.path.basename(video_path)}". Falling back to 30 FPS.')
            fps = 30.0
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_name = f'{output_dir}{os.sep}{basename}_{scene[0]}-{scene[1]}'
        output_ext = ".mp4"

        video.set(cv2.CAP_PROP_POS_FRAMES, (scene[1] + scene[0])//2)
        frame_number = int(video.get(cv2.CAP_PROP_POS_FRAMES))
        success, frame = video.read()
        if not success or frame is None:
            self.log("error", f'Failed to read frame from "{os.path.basename(video_path)}" at {int(frame_number)}. Skipping clip.')
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
            x1, y1, w1, h1 = self.__find_main_contour(frame_blend)
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
                    self.log("error", f"Failed to remove conversion placeholder {output_name + output_ext}, remove manually or check folder permissions.")

    def extract_images_multi(self, batch_mode : bool):
        cfg = self.bulk_read("images.output", "images.capture_rate", "images.blur_removal",
                             "images.crop_variation", "images.single_video", "images.directory",
                             "images.output_to_subdirectories", "images.directory", "images.remove_borders",
                             "images.range_start", "images.range_end",
                             as_dict=True)


        # if not pathlib.Path(cfg["images.output"]).is_dir() or cfg["images.output"] == "":
        #     self.log("error", "Invalid output directory!")
        #     return
        pathlib.Path(cfg["images.output"]).mkdir(parents=True, exist_ok=True)

        # validate numeric inputs
        try:
            capture_rate = float(cfg["images.capture_rate"])
            blur_threshold = float(cfg["images.blur_removal"])
            crop_variation = float(cfg["images.crop_variation"])
        except ValueError:
            self.log("error", "Invalid numeric input for Images/sec, Blur Removal, or Crop Variation.")
            return
        # if capture_rate <= 0:
        #     self.log("error", "Images/sec must be > 0.")
        #     return
        # if not (0.0 <= blur_threshold < 1.0):
        #     self.log("error", "Blur Removal must be between 0.0 and 1.0.")
        #     return
        # if not (0.0 <= crop_variation < 1.0):
        #     self.log("error", "Crop Variation must be between 0.0 and 1.0.")
        #     return

        input_videos = self.__get_vid_paths(batch_mode, cfg["images.single_video"], cfg["images.directory"])
        if len(input_videos) == 0:  #exit if no paths found
            return

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            for video_path in input_videos:
                if cfg["images.output_to_subdirectories"] and batch_mode:
                    output_directory = os.path.join(cfg["images.output"],
                                                    os.path.splitext(os.path.relpath(video_path, cfg["images.directory"]))[0])
                elif cfg["images.output_to_subdirectories"] and not batch_mode:
                    output_directory = os.path.join(cfg["images.output"],
                                                    os.path.splitext(os.path.basename(video_path))[0])
                else:
                    output_directory = cfg["images.output"]

                if batch_mode:
                    executor.submit(self.__save_frames,
                                    str(video_path), "00:00:00", "99:99:99", capture_rate,
                                    blur_threshold, cfg["images.remove_borders"], crop_variation, output_directory)
                else:
                    executor.submit(self.__save_frames,
                                    str(video_path), str(cfg["images.range_start"]), str(cfg["images.range_end"]), capture_rate,
                                    blur_threshold, cfg["images.remove_borders"], crop_variation, output_directory)
        if batch_mode:
            self.log("info", f'Image extraction from all videos in {cfg["images.directory"]} complete')
        else:
            self.log("info", f'Image extraction from {cfg["images.single_video"]} complete')

    def __save_frames(self, video_path: str, timestamp_min: str, timestamp_max: str, capture_rate: float,
                      blur_threshold: float, remove_borders : bool, crop_variation: float, output_dir: str):
        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS) or 0.0
        if fps <= 0:
            self.log("warning", f'Could not read FPS for "{os.path.basename(video_path)}". Falling back to 30 FPS.')
            fps = 30.0
        if capture_rate <= 0:
            self.log("error", "Images/sec must be > 0.")
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

        self.log("info", f'Video "{os.path.basename(video_path)}" will be split into {len(frame_list)} images in {output_dir}...')

        output_list = []
        for f in frame_list:
            video.set(cv2.CAP_PROP_POS_FRAMES, f)
            success, frame = video.read()
            if success and frame is not None:
                frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame_sharpness = cv2.Laplacian(frame_grayscale, cv2.CV_64F).var()  #get sharpness of greyscale pic
                output_list.append((f, frame_sharpness))

        if not output_list:
            self.log("warning", f'No frames extracted from {os.path.basename(video_path)} in the selected range.')
            video.release()
            return

        output_list_sorted = sorted(output_list, key=lambda x: x[1])
        cutoff = int(blur_threshold*len(output_list_sorted))     #calculate cutoff as portion of total frames
        output_list_cut = output_list_sorted[cutoff:]            # keep all frames above cutoff
        self.log("info", f'{cutoff} blurriest images have been dropped from {os.path.basename(video_path)}')

        basename, ext = os.path.splitext(os.path.basename(video_path))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for f in output_list_cut:
            filename = f'{output_dir}{os.sep}{basename}_{f[0]}.jpg'
            video.set(cv2.CAP_PROP_POS_FRAMES, f[0])
            success, frame = video.read()

            #crop out borders of frame
            if remove_borders and success and frame is not None:
                x1, y1, w1, h1 = self.__find_main_contour(frame)
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

    def download_multi(self, batch_mode: bool):
        cfg = self.bulk_read("download.output", "download.single_link",
                             "download.link_list", "download.additional_args",
                             as_dict=True)

        # if not pathlib.Path(cfg["download.output"]).is_dir() or cfg["download.output"] == "":
        #     self.log("error", "Invalid output directory!")
        #     return
        pathlib.Path(cfg["download.output"]).mkdir(parents=True, exist_ok=True)

        if not batch_mode:
            ydl_urls = [cfg["download.single_link"]]
        elif batch_mode:
            ydl_path = pathlib.Path(cfg["download.link_list"])
            if ydl_path.is_file() and ydl_path.suffix.lower() == ".txt":
                with open(ydl_path) as file:
                    ydl_urls = file.readlines()
            else:
                self.log("error", "Invalid link list!")
                return

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            for url in ydl_urls:
                executor.submit(self.__download_video,
                                url.strip(), cfg["download.output"], cfg["download.additional_args"])

        self.log("info", f'Completed {len(ydl_urls)} downloads.')

    def __download_video(self, url: str, output_dir: str, output_args: str):
        url = (url or "").strip()
        if not url:
            self.log("warning", "Empty URL, skipping download.")
            return

        additional_args = shlex.split(output_args.strip()) if output_args and output_args.strip() else [] # Respect quotes and split into list

        yt_dlp = shutil.which("yt-dlp")
        if yt_dlp is not None:
            cmd = [yt_dlp, "-o", "%(title)s.%(ext)s", "-P", output_dir] + additional_args + [url]

            self.log("info", f'Downloading {url}...')
            exitcode = subprocess.run(cmd).returncode
            if exitcode == 0:
                self.log("info", f'Download {url} done!')
            else:
                self.log("error", f'Failed to download {url} (process terminated with exit code {exitcode})')
        else:
            self.log("critical", 'yt-dlp executable not found in $PATH')
