import concurrent.futures
import logging
import math
import os
import pathlib
import random
import shlex
import subprocess
import threading
from typing import Literal

from web.backend.services._singleton import SingletonMixin

logger = logging.getLogger(__name__)

VideoStatus = Literal["idle", "running", "completed", "error"]

SUPPORTED_VIDEO_EXTENSIONS = {".webm", ".mkv", ".flv", ".avi", ".mov", ".wmv", ".mp4", ".mpeg", ".m4v"}

ALLOWED_YTDLP_FLAGS = {
    "--quiet",
    "-q",
    "--no-warnings",
    "--progress",
    "--no-progress",
    "--format",
    "-f",
    "--merge-output-format",
    "--write-subs",
    "--sub-lang",
    "--sub-format",
    "--embed-subs",
    "--no-playlist",
    "--playlist-items",
    "--no-overwrites",
    "--restrict-filenames",
    "--trim-filenames",
    "--recode-video",
    "--remux-video",
    "--write-thumbnail",
    "--convert-thumbnails",
}


class VideoService(SingletonMixin):
    def __init__(self) -> None:
        self._status: VideoStatus = "idle"
        self._message: str | None = None
        self._error: str | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def _set_status(self, status: VideoStatus, message: str | None = None, error: str | None = None) -> None:
        with self._lock:
            self._status = status
            self._message = message
            self._error = error

    def get_status(self) -> dict:
        with self._lock:
            return {
                "status": self._status,
                "message": self._message,
                "error": self._error,
            }

    @staticmethod
    def _import_cv2():
        try:
            import cv2

            return cv2
        except ImportError as exc:
            raise RuntimeError(
                "opencv-python (cv2) is required for video tools. Install it with:  pip install opencv-python"
            ) from exc

    @staticmethod
    def _import_scenedetect():
        try:
            import scenedetect

            return scenedetect
        except ImportError as exc:
            raise RuntimeError(
                "scenedetect is required for scene-based clip splitting. "
                "Install it with:  pip install scenedetect[opencv]"
            ) from exc

    def _get_vid_paths(self, batch_mode: bool, video_path: str, directory: str) -> list[pathlib.Path]:
        cv2 = self._import_cv2()
        input_videos: list[pathlib.Path] = []

        if not batch_mode:
            path = pathlib.Path(video_path)
            if not path.is_file():
                raise ValueError(f"No file specified, or invalid file path: {video_path}")
            vid = cv2.VideoCapture(str(path))
            ok = False
            try:
                if vid.isOpened():
                    ok, _ = vid.read()
            finally:
                vid.release()
            if not ok:
                raise ValueError(f"Invalid video file: {video_path}")
            return [path]
        else:
            if not directory or not pathlib.Path(directory).is_dir():
                raise ValueError(f"Invalid input directory: {directory}")
            lower_exts = {e.lower() for e in SUPPORTED_VIDEO_EXTENSIONS}
            for path in pathlib.Path(directory).rglob("*"):
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
            logger.info("Found %d videos to process", len(input_videos))
            return input_videos

    @staticmethod
    def _get_random_aspect(height: int, width: int, variation: float) -> tuple[int, int, int, int]:
        if variation == 0:
            return 0, height, 0, width

        old_aspect = height / width
        variation_scaled = old_aspect * variation
        if old_aspect > 1.2:
            new_aspect = min(
                4.0,
                max(
                    1.0,
                    random.triangular(
                        old_aspect - (variation_scaled * 1.5), old_aspect + (variation_scaled / 2), old_aspect
                    ),
                ),
            )
        elif old_aspect < 0.85:
            new_aspect = max(
                0.25,
                min(
                    1.0,
                    random.triangular(
                        old_aspect - (variation_scaled / 2), old_aspect + (variation_scaled * 1.5), old_aspect
                    ),
                ),
            )
        else:
            new_aspect = random.triangular(old_aspect - variation_scaled, old_aspect + variation_scaled)

        new_aspect = round(new_aspect, 2)
        if new_aspect > old_aspect:
            new_height = int(height)
            new_width = int(width * (old_aspect / new_aspect))
        elif new_aspect < old_aspect:
            new_height = int(height * (new_aspect / old_aspect))
            new_width = int(width)
        else:
            new_height = int(height)
            new_width = int(width)

        position_x = random.randint(0, max(width - new_width, 0))
        position_y = random.randint(0, max(height - new_height, 0))
        return position_y, new_height, position_x, new_width

    def _find_main_contour(self, frame):
        cv2 = self._import_cv2()
        frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, frame_thresh = cv2.threshold(frame_grayscale, 15, 255, cv2.THRESH_BINARY)
        frame_contours, _ = cv2.findContours(frame_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if frame_contours:
            frame_maincontour = max(frame_contours, key=cv2.contourArea)
            x1, y1, w1, h1 = cv2.boundingRect(frame_maincontour)
        else:
            x1 = 0
            y1 = 0
            h1, w1, _ = frame.shape
        if not frame_contours or h1 < 10 or w1 < 10:
            x1 = 0
            y1 = 0
            h1, w1, _ = frame.shape
        return x1, y1, w1, h1

    @staticmethod
    def _parse_timestamp_to_frames(timestamp: str, fps: float) -> int:
        parts = timestamp.split(":")
        seconds = sum(int(x) * 60**i for i, x in enumerate(reversed(parts)))
        return int(seconds * fps)

    def extract_clips(
        self,
        video_path: str,
        directory: str,
        batch_mode: bool,
        output_dir: str,
        time_start: str,
        time_end: str,
        output_subdirectories: bool,
        split_at_cuts: bool,
        max_length: float,
        fps: int,
        remove_borders: bool,
        crop_variation: float,
    ) -> dict:
        if not output_dir or not pathlib.Path(output_dir).is_dir():
            return {"ok": False, "error": f"Invalid output directory: {output_dir}"}
        if max_length <= 0.25:
            return {"ok": False, "error": "Max Length of clips must be > 0.25 seconds."}
        if fps < 0:
            return {"ok": False, "error": "Target FPS must be a positive integer (or 0 to skip fps re-encoding)."}
        if not (0.0 <= crop_variation < 1.0):
            return {"ok": False, "error": "Crop Variation must be between 0.0 and 1.0."}

        with self._lock:
            if self._status == "running":
                return {"ok": False, "error": "A video operation is already running"}
            self._status = "running"
            self._message = "Starting clip extraction..."
            self._error = None

        def _run():
            try:
                input_videos = self._get_vid_paths(batch_mode, video_path, directory)
                if len(input_videos) == 0:
                    self._set_status("error", error="No valid video files found.")
                    return

                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                    for vpath in input_videos:
                        if output_subdirectories and batch_mode:
                            out = os.path.join(output_dir, os.path.splitext(os.path.relpath(vpath, directory))[0])
                        elif output_subdirectories and not batch_mode:
                            out = os.path.join(output_dir, os.path.splitext(os.path.basename(vpath))[0])
                        else:
                            out = output_dir

                        ts_start = "00:00:00" if batch_mode else time_start
                        ts_end = "99:99:99" if batch_mode else time_end
                        executor.submit(
                            self._extract_clips_single,
                            str(vpath),
                            ts_start,
                            ts_end,
                            max_length,
                            split_at_cuts,
                            remove_borders,
                            crop_variation,
                            fps,
                            out,
                        )

                msg = (
                    f"Clip extraction from all videos in {directory} complete"
                    if batch_mode
                    else f"Clip extraction from {video_path} complete"
                )
                self._set_status("completed", message=msg)
                logger.info(msg)
            except Exception as exc:
                self._set_status("error", error=str(exc))
                logger.exception("Clip extraction failed")

        thread = threading.Thread(target=_run, daemon=True, name="video-extract-clips")
        self._thread = thread
        thread.start()
        return {"ok": True, "message": "Clip extraction started"}

    def _extract_clips_single(
        self,
        video_path: str,
        timestamp_min: str,
        timestamp_max: str,
        max_length: float,
        split_at_cuts: bool,
        remove_borders: bool,
        crop_variation: float,
        target_fps: int,
        output_dir: str,
    ) -> None:
        cv2 = self._import_cv2()
        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS) or 0.0
        if fps <= 0:
            logger.warning("Could not read FPS for '%s'. Falling back to 30 FPS.", os.path.basename(video_path))
            fps = 30.0

        max_length_frames = int(max_length * fps)
        min_length_frames = max(int(0.25 * fps), 1)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        timestamp_max_frame = self._parse_timestamp_to_frames(timestamp_max, fps)
        timestamp_max_frame = min(timestamp_max_frame, max(total_frames - 1, 0))
        timestamp_min_frame = self._parse_timestamp_to_frames(timestamp_min, fps)
        timestamp_min_frame = min(timestamp_min_frame, timestamp_max_frame)

        if split_at_cuts:
            scenedetect = self._import_scenedetect()
            timecode_list = scenedetect.detect(
                str(video_path),
                scenedetect.AdaptiveDetector(),
                start_time=int(timestamp_min_frame),
                end_time=int(timestamp_max_frame),
            )
            scene_list = [(x[0].get_frames(), x[1].get_frames()) for x in timecode_list]
            if len(scene_list) == 0:
                scene_list = [(timestamp_min_frame, timestamp_max_frame)]
        else:
            scene_list = [(timestamp_min_frame, timestamp_max_frame)]

        scene_list_split: list[tuple[int, int]] = []
        for scene in scene_list:
            length = scene[1] - scene[0]
            if length > max_length_frames:
                n = math.ceil(length / max_length_frames)
                new_length = int(length / n)
                new_splits = list(range(scene[0], scene[1] + min_length_frames, new_length))
                scene_list_split.extend(
                    (new_splits[i], new_splits[i + 1])
                    for i in range(len(new_splits) - 1)
                    if new_splits[i + 1] - new_splits[i] > min_length_frames
                )
            else:
                if length > (min_length_frames + 2):
                    scene_list_split.append((scene[0] + 1, scene[1] - 1))

        logger.info(
            "Video '%s' being split into %d clips in %s...",
            os.path.basename(video_path),
            len(scene_list_split),
            output_dir,
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            for scene in scene_list_split:
                executor.submit(
                    self._save_clip,
                    scene,
                    video_path,
                    target_fps,
                    remove_borders,
                    crop_variation,
                    output_dir,
                )

        video.release()

    def _save_clip(
        self,
        scene: tuple[int, int],
        video_path: str,
        target_fps: int,
        remove_borders: bool,
        crop_variation: float,
        output_dir: str,
    ) -> None:
        cv2 = self._import_cv2()
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        basename, _ext = os.path.splitext(os.path.basename(video_path))
        video = cv2.VideoCapture(str(video_path))
        fps = video.get(cv2.CAP_PROP_FPS) or 0.0
        if fps <= 0:
            fps = 30.0
        os.makedirs(output_dir, exist_ok=True)
        output_name = os.path.join(output_dir, f"{basename}_{scene[0]}-{scene[1]}")
        output_ext = ".mp4"

        video.set(cv2.CAP_PROP_POS_FRAMES, (scene[1] + scene[0]) // 2)
        success, frame = video.read()
        if not success or frame is None:
            logger.warning(
                "Failed to read frame from '%s' at midpoint. Skipping clip.",
                os.path.basename(video_path),
            )
            video.release()
            return

        if remove_borders:
            frame_blend = frame
            for i in range(5):
                random_frame = random.randint(scene[0], scene[1])
                video.set(cv2.CAP_PROP_POS_FRAMES, random_frame)
                success, frame = video.read()
                if not success or frame is None:
                    continue
                a = 1 / (i + 1)
                b = 1 - a
                frame_blend = cv2.addWeighted(frame, a, frame_blend, b, 0)
            x1, y1, w1, h1 = self._find_main_contour(frame_blend)
        else:
            x1 = 0
            y1 = 0
            h1, w1, _ = frame.shape

        y2, h2, x2, w2 = self._get_random_aspect(h1, w1, crop_variation)
        writer = cv2.VideoWriter(output_name + output_ext, fourcc, fps, (w2, h2))
        video.set(cv2.CAP_PROP_POS_FRAMES, scene[0])
        frame_number = int(video.get(cv2.CAP_PROP_POS_FRAMES))
        success, frame = video.read()

        while success and (frame_number < scene[1]):
            frame_trimmed = frame[y1 : y1 + h1, x1 : x1 + w1]
            writer.write(frame_trimmed[y2 : y2 + h2, x2 : x2 + w2])
            success, frame = video.read()
            frame_number += 1
        writer.release()
        video.release()

        if target_fps > 0:
            if int(round(fps)) == target_fps:
                return
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                f"{output_name}{output_ext}",
                "-filter:v",
                f"fps={target_fps}",
                "-an",
                f"{output_name}_{target_fps}fps{output_ext}",
            ]
            proc = subprocess.run(cmd, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            if proc.returncode == 0:
                try:
                    os.remove(output_name + output_ext)
                except OSError:
                    logger.warning("Failed to remove placeholder %s", output_name + output_ext)

    def extract_images(
        self,
        video_path: str,
        directory: str,
        batch_mode: bool,
        output_dir: str,
        time_start: str,
        time_end: str,
        output_subdirectories: bool,
        images_per_second: float,
        blur_removal: float,
        remove_borders: bool,
        crop_variation: float,
    ) -> dict:
        if not output_dir or not pathlib.Path(output_dir).is_dir():
            return {"ok": False, "error": f"Invalid output directory: {output_dir}"}
        if images_per_second <= 0:
            return {"ok": False, "error": "Images/sec must be > 0."}
        if not (0.0 <= blur_removal < 1.0):
            return {"ok": False, "error": "Blur Removal must be between 0.0 and 1.0."}
        if not (0.0 <= crop_variation < 1.0):
            return {"ok": False, "error": "Crop Variation must be between 0.0 and 1.0."}

        with self._lock:
            if self._status == "running":
                return {"ok": False, "error": "A video operation is already running"}
            self._status = "running"
            self._message = "Starting image extraction..."
            self._error = None

        def _run():
            try:
                input_videos = self._get_vid_paths(batch_mode, video_path, directory)
                if len(input_videos) == 0:
                    self._set_status("error", error="No valid video files found.")
                    return

                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                    for vpath in input_videos:
                        if output_subdirectories and batch_mode:
                            out = os.path.join(output_dir, os.path.splitext(os.path.relpath(vpath, directory))[0])
                        elif output_subdirectories and not batch_mode:
                            out = os.path.join(output_dir, os.path.splitext(os.path.basename(vpath))[0])
                        else:
                            out = output_dir

                        ts_start = "00:00:00" if batch_mode else time_start
                        ts_end = "99:99:99" if batch_mode else time_end
                        executor.submit(
                            self._save_frames,
                            str(vpath),
                            ts_start,
                            ts_end,
                            images_per_second,
                            blur_removal,
                            remove_borders,
                            crop_variation,
                            out,
                        )

                msg = (
                    f"Image extraction from all videos in {directory} complete"
                    if batch_mode
                    else f"Image extraction from {video_path} complete"
                )
                self._set_status("completed", message=msg)
                logger.info(msg)
            except Exception as exc:
                self._set_status("error", error=str(exc))
                logger.exception("Image extraction failed")

        thread = threading.Thread(target=_run, daemon=True, name="video-extract-images")
        self._thread = thread
        thread.start()
        return {"ok": True, "message": "Image extraction started"}

    def _save_frames(
        self,
        video_path: str,
        timestamp_min: str,
        timestamp_max: str,
        capture_rate: float,
        blur_threshold: float,
        remove_borders: bool,
        crop_variation: float,
        output_dir: str,
    ) -> None:
        cv2 = self._import_cv2()
        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS) or 0.0
        if fps <= 0:
            logger.warning("Could not read FPS for '%s'. Falling back to 30 FPS.", os.path.basename(video_path))
            fps = 30.0
        if capture_rate <= 0:
            video.release()
            return

        image_rate = max(int(fps / capture_rate), 1)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        timestamp_max_frame = self._parse_timestamp_to_frames(timestamp_max, fps)
        timestamp_max_frame = min(timestamp_max_frame, max(total_frames - 1, 0))
        timestamp_min_frame = self._parse_timestamp_to_frames(timestamp_min, fps)
        timestamp_min_frame = min(timestamp_min_frame, timestamp_max_frame)
        frame_range = range(timestamp_min_frame, timestamp_max_frame, image_rate)
        frame_list: list[int] = []

        for n in frame_range:
            frame_idx = abs(int(random.triangular(n - (image_rate / 2), n + (image_rate / 2))))
            frame_idx = max(0, min(frame_idx, max(total_frames - 1, 0)))
            frame_list.append(frame_idx)

        logger.info(
            "Video '%s' will be split into %d images in %s...",
            os.path.basename(video_path),
            len(frame_list),
            output_dir,
        )

        output_list: list[tuple[int, float]] = []
        for f in frame_list:
            video.set(cv2.CAP_PROP_POS_FRAMES, f)
            success, frame = video.read()
            if success and frame is not None:
                frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame_sharpness = cv2.Laplacian(frame_grayscale, cv2.CV_64F).var()
                output_list.append((f, frame_sharpness))

        if not output_list:
            logger.warning("No frames extracted from %s in the selected range.", os.path.basename(video_path))
            video.release()
            return

        output_list_sorted = sorted(output_list, key=lambda x: x[1])
        cutoff = int(blur_threshold * len(output_list_sorted))
        output_list_cut = output_list_sorted[cutoff:]
        logger.info("%d blurriest images have been dropped from %s", cutoff, os.path.basename(video_path))

        basename, _ext = os.path.splitext(os.path.basename(video_path))
        os.makedirs(output_dir, exist_ok=True)

        for f_idx, _sharpness in output_list_cut:
            filename = os.path.join(output_dir, f"{basename}_{f_idx}.jpg")
            video.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            success, frame = video.read()

            if remove_borders and success and frame is not None:
                x1, y1, w1, h1 = self._find_main_contour(frame)
                frame_cropped = frame[y1 : y1 + h1, x1 : x1 + w1]
            else:
                frame_cropped = frame if success and frame is not None else None
                if frame_cropped is not None:
                    x1 = 0
                    y1 = 0
                    h1, w1, _ = frame_cropped.shape

            if frame_cropped is not None:
                y2, h2, x2, w2 = self._get_random_aspect(h1, w1, crop_variation)
                cv2.imwrite(filename, frame_cropped[y2 : y2 + h2, x2 : x2 + w2])
        video.release()

    def download_videos(
        self,
        url: str,
        link_list_path: str,
        batch_mode: bool,
        output_dir: str,
        additional_args: str,
    ) -> dict:
        if not output_dir or not pathlib.Path(output_dir).is_dir():
            return {"ok": False, "error": f"Invalid output directory: {output_dir}"}

        if additional_args and additional_args.strip():
            try:
                self._validate_ytdlp_args(shlex.split(additional_args.strip()))
            except ValueError as exc:
                return {"ok": False, "error": str(exc)}

        if not batch_mode:
            if not url or not url.strip():
                return {"ok": False, "error": "No URL specified."}
            ydl_urls = [url.strip()]
        else:
            ydl_path = pathlib.Path(link_list_path)
            if not ydl_path.is_file() or ydl_path.suffix.lower() != ".txt":
                return {"ok": False, "error": f"Invalid link list file: {link_list_path}"}
            with open(ydl_path) as file:
                ydl_urls = [line.strip() for line in file.readlines() if line.strip()]
            if not ydl_urls:
                return {"ok": False, "error": "Link list file is empty."}

        with self._lock:
            if self._status == "running":
                return {"ok": False, "error": "A video operation is already running"}
            self._status = "running"
            self._message = f"Downloading {len(ydl_urls)} video(s)..."
            self._error = None

        def _run():
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                    for u in ydl_urls:
                        executor.submit(self._download_single, u, output_dir, additional_args)

                msg = f"Completed {len(ydl_urls)} download(s)."
                self._set_status("completed", message=msg)
                logger.info(msg)
            except Exception as exc:
                self._set_status("error", error=str(exc))
                logger.exception("Download failed")

        thread = threading.Thread(target=_run, daemon=True, name="video-download")
        self._thread = thread
        thread.start()
        return {"ok": True, "message": f"Downloading {len(ydl_urls)} video(s)..."}

    @staticmethod
    def _validate_ytdlp_args(args_list: list[str]) -> list[str]:
        rejected: list[str] = []
        for arg in args_list:
            flag = arg.split("=", 1)[0] if arg.startswith("-") else arg
            if flag.startswith("-") and flag not in ALLOWED_YTDLP_FLAGS:
                rejected.append(flag)
        if rejected:
            raise ValueError(
                f"yt-dlp flag(s) not allowed: {', '.join(rejected)}. "
                f"Only the following flags are permitted: {', '.join(sorted(ALLOWED_YTDLP_FLAGS))}"
            )
        return args_list

    @staticmethod
    def _download_single(url: str, output_dir: str, additional_args: str) -> None:
        url = (url or "").strip()
        if not url:
            logger.warning("Empty URL, skipping download.")
            return

        args_list = shlex.split(additional_args.strip()) if additional_args and additional_args.strip() else []

        try:
            VideoService._validate_ytdlp_args(args_list)
        except ValueError as exc:
            logger.error("Blocked yt-dlp arguments: %s", exc)
            return

        # "--" stops yt-dlp's argparse-based CLI from ever interpreting `url` as a flag.
        cmd = ["yt-dlp", "--ignore-config", "-o", "%(title)s.%(ext)s", "-P", output_dir] + args_list + ["--", url]

        logger.info("Downloading %s...", url)
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            logger.warning("yt-dlp exited %d for %s: %s", proc.returncode, url, proc.stderr[:500])
        logger.info("Download %s done!", url)
