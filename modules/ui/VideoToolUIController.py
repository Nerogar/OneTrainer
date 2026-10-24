import concurrent.futures
import math
import os
import pathlib
import random
import shlex
import subprocess
import threading
from fractions import Fraction

from modules.util.path_util import SUPPORTED_VIDEO_EXTENSIONS

import av
import cv2
import scenedetect
from PIL import Image


class VideoToolUIController:
    def __init__(self):
        self.view = None
        self.args = {
            "clip_single": "",
            "clip_list": "",
            "clip_time_start": "00:00:00",
            "clip_time_end": "99:99:99",
            "clip_output": "",
            "output_subdir_clip": False,
            "split_cuts": False,
            "clip_length": "3",
            "clip_fps": "24.0",
            "clip_bordercrop": False,
            "clip_crop": "0.2",
            "image_single": "",
            "image_list": "",
            "image_time_start": "00:00:00",
            "image_time_end": "99:99:99",
            "image_output": "",
            "output_subdir_img": False,
            "capture_rate": "0.5",
            "blur_threshold": "0.2",
            "image_bordercrop": False,
            "image_crop": "0.2",
            "download_link": "",
            "download_list": "",
            "download_output": "",
            "download_args": "--quiet --no-warnings --progress --format mp4",
        }

    def create_window(self, parent, view_cls):
        self.view = view_cls(parent, self)
        return self.view

    def __update_status(self, status_text: str):
        print(status_text)
        self.view.update_status(status_text)

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
                    self.__update_status("Invalid video file!")
                    return []
            else:
                self.__update_status("No file specified, or invalid file path!")
                return []
        else:
            input_videos = []
            if not pathlib.Path(input_path_dir).is_dir() or input_path_dir == "":
                self.__update_status("Invalid input directory!")
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
            self.__update_status(f'Found {len(input_videos)} videos to process')
            return input_videos

    def __run_in_thread(self, target, *args):
        """Clear status box and run target function in a daemon thread."""
        self.view.clear_status()
        t = threading.Thread(target=target, args=args)
        t.daemon = True
        t.start()

    @staticmethod
    def __parse_timestamp_to_frames(timestamp: str, fps: float) -> int:
        return int(sum(int(x) * 60 ** i for i, x in enumerate(reversed(timestamp.split(':')))) * fps)

    def __get_safe_fps(self, video: cv2.VideoCapture, video_path: str) -> float:
        fps = video.get(cv2.CAP_PROP_FPS) or 0.0
        if fps <= 0:
            self.__update_status(f'Warning: Could not read FPS for "{os.path.basename(video_path)}". Falling back to 30 FPS.')
            return 30.0
        return fps

    @staticmethod
    def __get_output_dir(use_subdir: bool, batch_mode: bool, output_entry: str,
                         video_path, input_dir: str) -> str:
        if use_subdir and batch_mode:
            return os.path.join(output_entry,
                                os.path.splitext(os.path.relpath(video_path, input_dir))[0])
        elif use_subdir:
            return os.path.join(output_entry,
                                os.path.splitext(os.path.basename(video_path))[0])
        return output_entry

    def __get_random_aspect(self, height: int, width: int, variation: float) -> tuple[int, int, int, int]:
        # Return original dimensions and no offset if variation is zero
        if variation == 0:
            return 0, height, 0, width

        old_aspect = height/width
        variation_scaled = old_aspect*variation
        if old_aspect > 1.2:        #tall image
            new_aspect = min(4.0, max(1.0, random.triangular(old_aspect-(variation_scaled*1.5), old_aspect+(variation_scaled/2), old_aspect)))
        elif old_aspect < 0.85:     #wide image
            new_aspect = max(0.25, min(1.0, random.triangular(old_aspect-(variation_scaled/2), old_aspect+(variation_scaled*1.5), old_aspect)))
        else:                       #square image
            new_aspect = random.triangular(old_aspect-variation_scaled, old_aspect+variation_scaled)

        new_aspect = round(new_aspect, 2)
        #keep the height the same if reducing width, and vice versa
        if new_aspect > old_aspect:
            new_height = int(height)
            new_width = int(width*(old_aspect/new_aspect))
        elif new_aspect < old_aspect:
            new_height = int(height*(new_aspect/old_aspect))
            new_width = int(width)
        else:
            new_height = int(height)
            new_width = int(width)

        #random offset in dimension that was cropped
        position_x = random.randint(0, width-new_width)
        position_y = random.randint(0, height-new_height)
        return position_y, new_height, position_x, new_width

    def find_main_contour(self, frame):
        #outline image to find main content and exclude black bars often present on letterboxed videos
        frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, frame_thresh = cv2.threshold(frame_grayscale, 15, 255, cv2.THRESH_BINARY)
        frame_contours, _ = cv2.findContours(frame_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if frame_contours:
            #select largest contour by area
            frame_maincontour = max(frame_contours, key=lambda c: cv2.contourArea(c))
            x1, y1, w1, h1 = cv2.boundingRect(frame_maincontour)
        else:   #fallback if no contours detected
            x1 = 0
            y1 = 0
            h1, w1, _ = frame.shape

        #if bounding box did not detect the correct area, likely due to all-black frame
        if not frame_contours or h1 < 10 or w1 < 10:
            x1 = 0
            y1 = 0
            h1, w1, _ = frame.shape
        return x1, y1, w1, h1

    def extract_clips_button(self, batch_mode: bool):
        self.__run_in_thread(self.__extract_clips_multi, batch_mode)

    def __extract_clips_multi(self, batch_mode: bool):
        p = self.args
        if not pathlib.Path(p['clip_output']).is_dir() or p['clip_output'] == "":
            self.__update_status("Invalid output directory!")
            return

        # validate numeric inputs
        try:
            max_length = float(p['clip_length'])
            crop_variation = float(p['clip_crop'])
            target_fps = float(p['clip_fps'])
        except ValueError:
            self.__update_status("Invalid numeric input for Max Length, Crop Variation, or FPS.")
            return
        if max_length <= 0.25:
            self.__update_status("Max Length of clips must be > 0.25 seconds.")
            return
        if target_fps < 0:
            self.__update_status("Target FPS must be a positive number (or 0 to skip fps re-encoding).")
            return
        if not (0.0 <= crop_variation < 1.0):
            self.__update_status("Crop Variation must be between 0.0 and 1.0.")
            return

        input_videos = self.__get_vid_paths(batch_mode, p['clip_single'], p['clip_list'])
        if len(input_videos) == 0:  # exit if no paths found
            return

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            for video_path in input_videos:
                output_directory = self.__get_output_dir(
                    p['output_subdir_clip'], batch_mode,
                    p['clip_output'], video_path, p['clip_list'])
                time_start = "00:00:00" if batch_mode else p['clip_time_start']
                time_end = "99:99:99" if batch_mode else p['clip_time_end']
                executor.submit(self.__extract_clips,
                                str(video_path), time_start, time_end, max_length,
                                p['split_cuts'], p['clip_bordercrop'],
                                crop_variation, target_fps, output_directory)

        if batch_mode:
            self.__update_status(f'Clip extraction from all videos in "{p["clip_list"]}" complete')
        else:
            self.__update_status(f'Clip extraction from "{p["clip_single"]}" complete')

    def __extract_clips(self, video_path: str, timestamp_min: str, timestamp_max: str, max_length: float,
                        split_at_cuts: bool, remove_borders: bool, crop_variation: float, target_fps: float, output_dir: str):
        video = cv2.VideoCapture(video_path)
        vid_fps = self.__get_safe_fps(video, video_path)
        max_length_frames = int(max_length * vid_fps)
        min_length_frames = max(int(0.25 * vid_fps), 1)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        timestamp_max_frame = min(self.__parse_timestamp_to_frames(timestamp_max, vid_fps), max(total_frames - 1, 0))
        timestamp_min_frame = min(self.__parse_timestamp_to_frames(timestamp_min, vid_fps), timestamp_max_frame)

        if split_at_cuts:
            #use scenedetect to find cuts, based on start/end frame number
            self.__update_status(f'Detecting scenes in "{os.path.basename(video_path)}"')
            timecode_list = scenedetect.detect(
                video_path=str(video_path),
                detector=scenedetect.AdaptiveDetector(),
                start_time=int(timestamp_min_frame),
                end_time=int(timestamp_max_frame))
            scene_list = [(x[0].get_frames(), x[1].get_frames()) for x in timecode_list]
            if not scene_list:
                scene_list = [(timestamp_min_frame, timestamp_max_frame)]
        else:
            scene_list = [(timestamp_min_frame, timestamp_max_frame)]

        scene_list_split = []
        for scene in scene_list:
            length = scene[1]-scene[0]
            if length > max_length_frames:  #check for any scenes longer than max length
                n = math.ceil(length/max_length_frames) #divide into n new scenes
                new_length = int(length/n)
                new_splits = range(scene[0], scene[1]+min_length_frames, new_length)   #divide clip into closest chunks to max_length
                for i, _n in enumerate(new_splits[:-1]):
                    if new_splits[i + 1] - new_splits[i] > min_length_frames:
                        scene_list_split.append((new_splits[i], new_splits[i + 1]))
            elif length > (min_length_frames + 2):
                # Trim first/last frame to avoid transition artifacts
                scene_list_split.append((scene[0] + 1, scene[1] - 1))

        self.__update_status(f'Video "{os.path.basename(video_path)}" being split into {len(scene_list_split)} clips in "{output_dir}"')

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(self.__save_clip, scene, video_path, target_fps,
                                remove_borders, crop_variation, output_dir)
                for scene in scene_list_split
            ]
            for future in concurrent.futures.as_completed(futures):
                exc = future.exception()
                if exc is not None:
                    self.__update_status(f'Error saving clip: {exc}')

        video.release()

    def __save_clip(self, scene: tuple[int, int], video_path: str, target_fps: float,
                    remove_borders: bool, crop_variation: float, output_dir: str):
        basename, ext = os.path.splitext(os.path.basename(video_path))
        video = cv2.VideoCapture(str(video_path))
        fps = self.__get_safe_fps(video, video_path)
        os.makedirs(output_dir, exist_ok=True)
        output_name = f'{output_dir}{os.sep}{basename}_{scene[0]}-{scene[1]}'
        output_ext = ".mp4"

        video.set(cv2.CAP_PROP_POS_FRAMES, (scene[1] + scene[0])//2)    #set to middle of scene
        frame_number = int(video.get(cv2.CAP_PROP_POS_FRAMES))
        success, frame = video.read()
        if not success or frame is None:
            self.__update_status(f'Failed to read frame from "{os.path.basename(video_path)}" at {int(frame_number)}. Skipping clip.')
            video.release()
            return

        # Blend random frames to detect borders, avoiding incorrect crop from black frames
        if remove_borders:
            frame_blend = frame
            for i in range(5):
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
        # Ensure dimensions are even, required
        h2 -= h2 % 2
        w2 -= w2 % 2
        print(end='\x1b[2K')    #clear terminal so next line can overwrite it
        print(f'Saving frames {scene[0]}-{scene[1]} at size {w2}x{h2}', end="\r")
        video.set(cv2.CAP_PROP_POS_FRAMES, (scene[1] + scene[0])//2)
        success, frame = video.read()
        if success:
            try:
                preview = Image.fromarray(
                        cv2.cvtColor(frame[y1+y2:y1+y2+h2, x1+x2:x1+x2+w2], cv2.COLOR_BGR2RGB))
                preview.thumbnail((150, 150))
                #truncate filename of long files so UI doesn't shift around
                filename_truncated = basename + ext if len(basename) < 20 else basename[:18] + ".." + ext
                self.view.update_preview(preview, f'{filename_truncated}\nFrames: {scene[0]}-{scene[1]}\nSize: {w2}x{h2}')
            except Exception:
                pass
        video.release()

        if target_fps <= 0:
            target_fps = fps

        output_path = f'{output_name}{output_ext}'
        self.__write_clip_av(video_path, output_path, scene, fps, target_fps,
                             x1 + x2, y1 + y2, w2, h2)

    @staticmethod
    def __write_clip_av(video_path: str, output_path: str, scene: tuple[int, int],
                        src_fps: float, target_fps: float,
                        crop_x: int, crop_y: int, crop_w: int, crop_h: int):
        start_sec = scene[0] / src_fps
        end_sec = scene[1] / src_fps
        rate_frac = Fraction(target_fps).limit_denominator(10000)
        stream_time_base = Fraction(rate_frac.denominator, rate_frac.numerator)

        with av.open(video_path) as input_container:
            in_video = input_container.streams.video[0]
            in_video.thread_type = 'AUTO'
            in_audio = input_container.streams.audio[0] if input_container.streams.audio else None

            with av.open(output_path, mode='w') as output_container:
                out_video = output_container.add_stream('libx264', rate=rate_frac)
                out_video.width = crop_w
                out_video.height = crop_h
                out_video.pix_fmt = 'yuv420p'
                out_video.time_base = stream_time_base

                out_audio = output_container.add_stream_from_template(in_audio) if in_audio else None

                input_container.seek(int(start_sec * 1_000_000))

                out_frame_idx = 0
                out_time_step = 1.0 / target_fps
                video_done = False
                decode_streams = [s for s in (in_video, in_audio) if s is not None]

                for packet in input_container.demux(decode_streams):
                    if packet.stream == in_video:
                        if video_done:
                            continue
                        for frame in packet.decode():
                            if frame.time is None or frame.time < start_sec:
                                continue
                            if frame.time >= end_sec:
                                video_done = True
                                break

                            # FPS conversion: skip frames when source fps > target fps
                            if frame.time < start_sec + out_frame_idx * out_time_step:
                                continue

                            img = frame.to_ndarray(format='bgr24')
                            cropped = img[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
                            out_frame = av.VideoFrame.from_ndarray(cropped, format='bgr24')
                            out_frame.pts = out_frame_idx
                            out_frame.time_base = stream_time_base

                            for out_pkt in out_video.encode(out_frame):
                                output_container.mux(out_pkt)
                            out_frame_idx += 1

                    elif packet.stream == in_audio and out_audio is not None:
                        if packet.dts is None:
                            continue
                        pkt_time = float(packet.pts * packet.time_base)
                        if pkt_time < start_sec or pkt_time >= end_sec:
                            continue
                        # Re-timestamp audio relative to clip start
                        packet.pts = int((pkt_time - start_sec) / packet.time_base)
                        packet.dts = packet.pts
                        packet.stream = out_audio
                        output_container.mux(packet)

                # Flush video encoder
                for pkt in out_video.encode():
                    output_container.mux(pkt)

    def extract_images_button(self, batch_mode: bool):
        self.__run_in_thread(self.__extract_images_multi, batch_mode)

    def __extract_images_multi(self, batch_mode: bool):
        p = self.args
        if not pathlib.Path(p['image_output']).is_dir() or p['image_output'] == "":
            self.__update_status("Invalid output directory!")
            return

        # validate numeric inputs
        try:
            capture_rate = float(p['capture_rate'])
            blur_threshold = float(p['blur_threshold'])
            crop_variation = float(p['image_crop'])
        except ValueError:
            self.__update_status("Invalid numeric input for Images/sec, Blur Removal, or Crop Variation.")
            return
        if capture_rate <= 0:
            self.__update_status("Images/sec must be > 0.")
            return
        if not (0.0 <= blur_threshold < 1.0):
            self.__update_status("Blur Removal must be between 0.0 and 1.0.")
            return
        if not (0.0 <= crop_variation < 1.0):
            self.__update_status("Crop Variation must be between 0.0 and 1.0.")
            return

        input_videos = self.__get_vid_paths(batch_mode, p['image_single'], p['image_list'])
        if not input_videos:
            return

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            for video_path in input_videos:
                output_directory = self.__get_output_dir(
                    p['output_subdir_img'], batch_mode,
                    p['image_output'], video_path, p['image_list'])
                time_start = "00:00:00" if batch_mode else p['image_time_start']
                time_end = "99:99:99" if batch_mode else p['image_time_end']
                executor.submit(self.__save_frames,
                                str(video_path), time_start, time_end, capture_rate,
                                blur_threshold, p['image_bordercrop'],
                                crop_variation, output_directory)
        if batch_mode:
            self.__update_status(f'Image extraction from all videos in {p["image_list"]} complete')
        else:
            self.__update_status(f'Image extraction from "{p["image_single"]}" complete')

    def __save_frames(self, video_path: str, timestamp_min: str, timestamp_max: str, capture_rate: float,
                      blur_threshold: float, remove_borders: bool, crop_variation: float, output_dir: str):
        video = cv2.VideoCapture(video_path)
        vid_fps = self.__get_safe_fps(video, video_path)
        if capture_rate <= 0:
            self.__update_status("Images/sec must be > 0.")
            video.release()
            return
        image_rate = max(int(vid_fps / capture_rate), 1)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        timestamp_max_frame = min(self.__parse_timestamp_to_frames(timestamp_max, vid_fps), max(total_frames - 1, 0))
        timestamp_min_frame = min(self.__parse_timestamp_to_frames(timestamp_min, vid_fps), timestamp_max_frame)
        frame_range = range(timestamp_min_frame, timestamp_max_frame, image_rate)
        frame_list = []

        for n in frame_range:
            #pick frame from random triangular distribution around center of each "chunk" of the video
            frame = abs(int(random.triangular(n-(image_rate/2), n+(image_rate/2))))
            frame = max(0, min(frame, max(total_frames - 1, 0)))
            frame_list.append(frame)

        self.__update_status(f'Video "{os.path.basename(video_path)}" will be split into {len(frame_list)} images in "{output_dir}"')

        output_list = []
        for f in frame_list:
            video.set(cv2.CAP_PROP_POS_FRAMES, f)
            success, frame = video.read()
            if success and frame is not None:
                frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame_sharpness = cv2.Laplacian(frame_grayscale, cv2.CV_64F).var()
                output_list.append((f, frame_sharpness))

        if not output_list:
            self.__update_status(f'No frames extracted from "{os.path.basename(video_path)}" in the selected range.')
            video.release()
            return

        output_list_sorted = sorted(output_list, key=lambda x: x[1])
        cutoff = int(blur_threshold * len(output_list_sorted))
        output_list_cut = output_list_sorted[cutoff:]
        self.__update_status(f'{cutoff} blurriest images have been dropped from "{os.path.basename(video_path)}"')

        basename, ext = os.path.splitext(os.path.basename(video_path))
        os.makedirs(output_dir, exist_ok=True)

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

            if success and frame is not None and frame_cropped is not None:
                print(end='\x1b[2K')    #clear terminal so next line can overwrite it
                print(f'Saving frame {f[0]} at size {w2}x{h2}', end="\r")
                try:
                    preview = Image.fromarray(
                        cv2.cvtColor(frame_cropped[y2:y2+h2, x2:x2+w2], cv2.COLOR_BGR2RGB))
                    preview.thumbnail((150, 150))
                    filename_truncated = basename + ext if len(basename) < 20 else basename[:17] + "..." + ext
                    self.view.update_preview(preview, f'{filename_truncated}\nFrame: {f[0]}\nSize: {w2}x{h2}')
                except Exception:
                    pass  # preview update is non-critical

                cv2.imwrite(filename, frame_cropped[y2:y2+h2, x2:x2+w2])
        video.release()

    def download_button(self, batch_mode: bool):
        self.__run_in_thread(self.__download_multi, batch_mode)

    def __download_multi(self, batch_mode: bool):
        p = self.args
        if not pathlib.Path(p['download_output']).is_dir() or p['download_output'] == "":
            self.__update_status("Invalid output directory!")
            return

        if not batch_mode:
            ydl_urls = [p['download_link']]
        elif batch_mode:
            ydl_path = pathlib.Path(p['download_list'])
            if ydl_path.is_file() and ydl_path.suffix.lower() == ".txt":
                with open(ydl_path) as file:
                    ydl_urls = file.readlines()
            else:
                self.__update_status("Invalid link list!")
                return

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            for url in ydl_urls:
                executor.submit(self.__download_video,
                                url.strip(), p['download_output'],
                                p['download_args'])

        self.__update_status(f'Completed {len(ydl_urls)} downloads.')

    def __download_video(self, url: str, output_dir: str, output_args: str):
        url = (url or "").strip()
        if not url:
            self.__update_status("Empty URL, skipping download.")
            return

        #Respect quotes and split into list to run as yt-dlp command
        additional_args = shlex.split(output_args.strip()) if output_args and output_args.strip() else []
        cmd = ["yt-dlp", "-o", "%(title)s.%(ext)s", "-P", output_dir] + additional_args + [url]

        self.__update_status(f'Downloading {url}')
        subprocess.run(cmd)
        self.__update_status(f'Download {url} done!')
