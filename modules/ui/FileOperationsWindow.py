"""
OneTrainer Image Tools Dialog

This module defines a CTkToplevel window for performing common batch operations on files
such as renaming, resizing, and format conversion.

Note on _verify_single_image:
  The image is opened twice because Pillow's .verify() invalidates the image object.
  Therefore, a second open (and .load()) is required to perform additional operations.
"""

import collections
import contextlib
import logging
import os
import random
import threading
import time
import tkinter as tk
from collections.abc import Callable  # Updated import
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue
from tkinter import filedialog
from typing import TypedDict

from modules.util import image_util, path_util
from modules.util.ui.ui_utils import (
    set_window_icon,
)
from modules.util.ui.UIState import UIState

import customtkinter as ctk
import imagesize
import oxipng
from PIL import Image, ImageColor

logger = logging.getLogger(__name__)
for handler in logging.root.handlers:
    handler.setFormatter(logging.Formatter("%(message)s"))

class Group(TypedDict):
    caption: Path | None
    masks: list[tuple[Path, str]]


# --- Constants ---
# Progress increments for sequential renaming and other operations
PROGRESS_RENAME_TEMP_START: float = 0.1
PROGRESS_RENAME_TEMP_RANGE: float = 0.4
PROGRESS_RENAME_FINAL_START: float = 0.5
PROGRESS_RENAME_FINAL_RANGE: float = 0.4

# Megapixel constants
ONE_MEGAPIXEL: int = 1_048_576
COMPUTE_PROOF_MEGAPIXEL_THRESHOLD: int = 4_194_304
MIDDLEGROUND_MEGAPIXEL_THRESHOLD: int = 8_388_608
FUTURE_PROOF_MEGAPIXEL_THRESHOLD: int = 16_777_216


class FileProcessor:
    """Handles the backend file processing operations."""

    __slots__ = ("config_data", "message_queue", "cancel_requested", "max_workers")

    def __init__(
        self,
        config_data: dict,
        message_queue: Queue,
        cancel_requested_func: Callable[[], bool],
        max_workers: int,
    ):
        self.config_data = config_data
        self.message_queue = message_queue
        self.cancel_requested = cancel_requested_func
        self.max_workers = max_workers

    def _log(self, message: str) -> None:
        """Queue a log message to be processed on the main thread."""
        self.message_queue.put(("log", message))

    def _update_status(
        self, message: str, progress: float | None = None
    ) -> None:
        """Queue a status update to be processed on the main thread."""
        self.message_queue.put(("status", message, progress))

    @staticmethod
    def calculate_dimensions_for_megapixels(
        original_width: int, original_height: int, target_pixels: int
    ) -> tuple[int, int]:
        """Calculates new dimensions to fit an image within a pixel budget."""
        original_pixels = original_width * original_height
        if original_pixels <= target_pixels:
            return original_width, original_height

        scale_factor = (target_pixels / original_pixels) ** 0.5
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)
        return new_width, new_height

    @staticmethod
    def _filter_is_image(f: Path) -> bool:
        """Filter for supported image files."""
        return path_util.is_supported_image_extension(f.suffix)

    @staticmethod
    def _filter_images_and_skip_masks(f: Path) -> bool:
        """Filter for supported image files, excluding mask files."""
        return path_util.is_supported_image_extension(f.suffix) and not f.stem.endswith("-masklabel")

    def _run_parallel_task(
        self,
        task_name: str,
        files: list[Path],
        worker_func: Callable[[Path], tuple[str, object | None]],
        summary_func: Callable[[dict[str, int], dict[str, list]], str],
        progress_callback: Callable[[float, str], None] | None = None,
        filter_func: Callable[[Path], bool] | None = None,
    ) -> dict:
        """
        Generic parallel task executor for image files.
        Handles filtering, threading, progress, cancellation, and logging.
        Returns a dictionary with the aggregated results.
        """
        if filter_func is None:
            filter_func = self._filter_is_image

        image_files = [f for f in files if filter_func(f)]
        self._log(f"{task_name}: Found {len(image_files)} files to process.")
        if not image_files:
            return {}

        results = collections.defaultdict(int)
        extra_data = collections.defaultdict(list)
        results_lock = threading.Lock()
        total_files = len(image_files)

        def process_wrapper(file: Path) -> None:
            if self.cancel_requested():
                return

            try:
                status, data = worker_func(file)
            except Exception as e:
                self._log(f"Error processing {file.name} during {task_name}: {e}")
                status, data = "errors", (file.name, str(e))

            with results_lock:
                results[status] += 1
                if data is not None:
                    extra_data[status].append(data)

                processed_count = sum(results.values())
                if progress_callback:
                    progress_callback(
                        (processed_count / total_files) * 100,
                        f"{task_name}: {file.name}",
                    )

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            list(executor.map(process_wrapper, image_files))

        if self.cancel_requested():
            self._log(f"{task_name} was cancelled.")
            return results

        summary_message = summary_func(results, extra_data)
        self._log(summary_message)

        if "errors" in extra_data:
            self._log("Problematic files:")
            for name, error in extra_data["errors"]:
                self._log(f"  - {name}: {error}")

        return results

    def verify_single_image(self, file: Path) -> None:
        """
        Verify a single image file for corruption. Raises ValueError on failure.

        Note: The image is opened twice because Pillow's .verify() invalidates the image object.
        """
        file_path = Path(file)
        try:
            with Image.open(file_path) as img:
                img.verify()
            with Image.open(file_path) as img:
                img.load()
                if hasattr(img, "getpixel"):
                    img.getpixel((0, 0))
        except Exception as e:
            raise ValueError(f"Image file {file_path.name} is corrupt or invalid: {e}") from e

    def verify_images(
        self,
        files: list[Path],
        progress_callback: Callable[[float, str | None], None] | None = None,
    ) -> dict:
        """Verify images for corruption using parallel processing."""
        self._log("Starting image verification process...")

        def worker(file: Path) -> tuple[str, object | None]:
            try:
                self.verify_single_image(file)
                self._log(f"✓ {file.name} is valid")
                return "valid", None
            except ValueError as e:
                original_error = e.__cause__ or e
                error_msg = str(original_error)
                self._log(f"✗ {file.name} is CORRUPTED: {error_msg}")
                return "errors", (file.name, error_msg)

        def summary(results: dict, extra_data: dict) -> str:
            valid_count = results.get("valid", 0)
            error_count = results.get("errors", 0)
            if error_count == 0:
                return f"✓ Image verification complete: All {valid_count} images are valid"

            return f"⚠ Image verification complete: {valid_count} valid, {error_count} corrupted/problematic"

        return self._run_parallel_task(
            task_name="Image Verification",
            files=files,
            worker_func=worker,
            summary_func=summary,
            progress_callback=progress_callback,
        )

    def convert_image_format(
        self,
        files: list[Path],
        target_format: str,
        skip_extensions: set,
        format_options: dict,
        progress_callback: Callable[[float, str | None], None] | None = None,
    ) -> None:
        """Generic image conversion function for multiple formats."""
        self._log(f"Starting conversion to {target_format} format...")

        # Use local variables for performance and clarity
        format_ext = format_options["format_ext"]
        pil_format = format_options["pil_format"]
        lossless_extensions = format_options.get("lossless_extensions", set())
        is_lossless_check = format_options.get("is_lossless_check", lambda f, i: False)
        quality = format_options.get("quality", 90)
        save_kwargs_base = format_options.get("save_kwargs", {})

        def file_filter(f: Path) -> bool:
            return (
                path_util.is_supported_image_extension(f.suffix)
                and f.suffix.lower() not in skip_extensions
                and "-" not in f.stem  # Skip mask files
            )

        def worker(file: Path) -> tuple[str, object | None]:
            original_size = file.stat().st_size
            new_path = file.with_suffix(format_ext)

            with Image.open(file) as img:
                is_lossless = file.suffix.lower() in lossless_extensions or is_lossless_check(file, img)

                save_kwargs = save_kwargs_base.copy()
                save_kwargs["quality"] = quality
                if is_lossless:
                    save_kwargs["lossless"] = True

                img.save(new_path, pil_format, **save_kwargs)

            if not new_path.exists():
                return "errors", (file.name, "Failed to save new file.")

            new_size = new_path.stat().st_size
            bytes_saved = original_size - new_size

            if new_size < original_size:
                file.unlink()
                self._log(f"Converted {file.name} to {target_format}: {bytes_saved:,} bytes saved")
                return "success", bytes_saved
            else:
                new_path.unlink()
                self._log(f"Skipped {file.name}: {target_format} version would be larger.")
                return "skipped", None

        def summary(results: dict, extra_data: dict) -> str:
            success_count = results.get("success", 0)
            total_bytes_saved = sum(extra_data.get("success", []))

            msg = f"Completed {target_format} conversion: {success_count} converted, {results.get('skipped', 0)} skipped, {results.get('errors', 0)} errors."
            if success_count > 0:
                avg_saving = total_bytes_saved / success_count
                msg += f"\nTotal bytes saved: {total_bytes_saved:,} (avg: {avg_saving:,.1f} per file)"
            return msg

        self._run_parallel_task(
            task_name=f"Convert to {target_format}",
            files=files,
            worker_func=worker,
            summary_func=summary,
            progress_callback=progress_callback,
            filter_func=file_filter,
        )

    def convert_to_webp(
        self,
        files: list[Path],
        progress_callback: Callable[[float, str | None], None]
        | None = None,
    ) -> None:
        """Convert images to WebP format using the generic converter."""
        format_options = {
            "format_ext": ".webp",
            "pil_format": "WEBP",
            "lossless_extensions": {".png", ".tiff", ".tif", ".bmp"},
            "is_lossless_check": lambda file, img: False,
            "quality": 90,
        }
        skip_extensions = {".webp", ".jxl", ".avif"}
        self.convert_image_format(
            files,
            "WebP",
            skip_extensions,
            format_options,
            progress_callback,
        )

    def convert_to_jpegxl(
        self,
        files: list[Path],
        progress_callback: Callable[[float, str | None], None]
        | None = None,
    ) -> None:
        """Convert images to JPEG XL format using the generic converter."""
        format_options = {
            "format_ext": ".jxl",
            "pil_format": "JXL",
            "lossless_extensions": set(),
            "is_lossless_check": lambda file, img: file.suffix.lower()
            in {".jpg", ".jpeg"},
            "quality": 90,
        }
        skip_extensions = {".jxl"}
        self.convert_image_format(
            files,
            "JPEG XL",
            skip_extensions,
            format_options,
            progress_callback,
        )

    def rename_files_sequentially(
        self,
        files: list[Path],
        progress_callback: Callable[[float, str | None], None] | None = None,
    ) -> list[Path]:
        """Rename files sequentially using a transactional approach for error resilience."""
        import uuid
        from collections import defaultdict

        def _safe_rename(src: Path, dst: Path) -> bool:
            """Attempt src → dst; log on predictable FS errors, bubble up anything else."""
            try:
                src.rename(dst)
                return True
            except OSError as exc:
                self._log(f"rename {src.name} → {dst.name} failed: {exc}")
                return False

        self._log("Starting sequential file renaming…")

        # --- File grouping ---
        groups = defaultdict(lambda: {"image": None, "caption": None, "masks": []})
        for f in files:
            stem = f.stem
            if stem.endswith("-masklabel"):
                base_stem = stem.removesuffix("-masklabel")
                groups[base_stem]["masks"].append((f, "masklabel"))
            elif path_util.is_supported_image_extension(f.suffix):
                groups[stem]["image"] = f
            elif f.suffix.lower() == ".txt":
                groups[stem]["caption"] = f

        # Filter for groups that actually have an image file and get the image paths
        image_groups = {stem: data for stem, data in groups.items() if data.get("image")}
        image_files = [data["image"] for data in image_groups.values()]

        self._log(
            f"Found {len(image_files)} images and their associated files to rename."
        )

        sorted_imgs = sorted(image_files, key=lambda p: p.name)
        if not sorted_imgs:
            progress_callback and progress_callback(1.0, "No image files found.")
            return []

        # Check if files are already sequential
        seq_needed = any(
            (not f.stem.isdigit()) or (int(f.stem) != i + 1)
            for i, f in enumerate(sorted_imgs)
        )
        if not seq_needed:
            self._log("Files are already named sequentially. No action needed.")
            progress_callback and progress_callback(1.0, "Files are already sequential.")
            return sorted_imgs

        # plan
        uid = uuid.uuid4().hex
        rename_plan: list[tuple[Path, Path, Path]] = []
        for i, img in enumerate(sorted_imgs, start=1):
            tmp = img.with_name(f"tmp_{uid}_{i}{img.suffix}")
            fin = img.with_name(f"{i}{img.suffix}")
            rename_plan.append((img, tmp, fin))

            grp = image_groups.get(img.stem)
            if grp:
                if (cap := grp.get("caption")) is not None:
                    rename_plan.append(
                        (
                            cap,
                            cap.with_name(f"tmp_{uid}_{i}.txt"),
                            cap.with_name(f"{i}.txt"),
                        )
                    )
                for m, label in grp.get("masks", []):
                    rename_plan.append(
                        (
                            m,
                            m.with_name(f"tmp_{uid}_{i}-{label}{m.suffix}"),
                            m.with_name(f"{i}-{label}{m.suffix}"),
                        )
                    )

        # -------------------------------------------------------- phase 1 → tmp
        succeeded: list[tuple[Path, Path]] = []
        total = len(rename_plan)
        for idx, (orig, tmp, _) in enumerate(rename_plan):
            progress_callback and progress_callback(
                PROGRESS_RENAME_TEMP_START + PROGRESS_RENAME_TEMP_RANGE * idx / total,
                f"Temp rename: {orig.name}",
            )
            if _safe_rename(orig, tmp):
                succeeded.append((orig, tmp))
            else:   # rollback every successful temp
                for o, t in reversed(succeeded):
                    _safe_rename(t, o)
                progress_callback and progress_callback(1.0, "Rename failed, rolled back.")
                return sorted_imgs

        # ------------------------------------------------------- phase 2 tmp → fin
        final_imgs: list[Path] = []
        for idx, (_, tmp, fin) in enumerate(rename_plan):
            progress_callback and progress_callback(
                PROGRESS_RENAME_FINAL_START + PROGRESS_RENAME_FINAL_RANGE * idx / total,
                f"Final rename: {tmp.name}",
            )
            if _safe_rename(tmp, fin):
                if path_util.is_supported_image_extension(fin.suffix) and not fin.stem.endswith("-masklabel"):
                    final_imgs.append(fin)

        progress_callback and progress_callback(1.0, "Renaming complete.")
        self._log(f"Sequential renaming complete: {len(final_imgs)} images renamed.")
        return sorted(final_imgs, key=lambda p: int(p.stem))

    def resize_large_images(
        self,
        files: list[Path],
        progress_callback: Callable[[float, str | None], None] | None = None,
    ) -> None:
        """Resize images larger than the defined threshold using a PIL-based pipeline."""
        cfg = self.config_data
        megapixels_map = {
            "1MP": ONE_MEGAPIXEL,
            "Compute Proof (4MP)": COMPUTE_PROOF_MEGAPIXEL_THRESHOLD,
            "Middleground (8MP)": MIDDLEGROUND_MEGAPIXEL_THRESHOLD,
            "Zoom-in proof(16MP)": FUTURE_PROOF_MEGAPIXEL_THRESHOLD,
        }
        resize_option = cfg["resize_megapixels"]
        if resize_option == "Custom":
            try:
                custom_mp = int(cfg["resize_custom_megapixels"])
                target_pixels = custom_mp * ONE_MEGAPIXEL
            except (ValueError, TypeError):
                self._log(
                    f"Invalid custom megapixel value '{cfg['resize_custom_megapixels']}'. Using 4MP."
                )
                target_pixels = COMPUTE_PROOF_MEGAPIXEL_THRESHOLD
        else:
            target_pixels = megapixels_map.get(
                resize_option, COMPUTE_PROOF_MEGAPIXEL_THRESHOLD
            )

        self._log(
            f"Starting resizing of large images... Target: {target_pixels / ONE_MEGAPIXEL:.1f}MP"
        )

        def worker(file: Path) -> tuple[str, object | None]:
            width, height = imagesize.get(file)
            if width * height <= target_pixels:
                return "skipped", None

            new_width, new_height = self.calculate_dimensions_for_megapixels(
                width, height, target_pixels
            )
            reduction_factor = width / new_width

            with Image.open(file) as img:
                resample_filter = (
                    Image.Resampling.LANCZOS
                    if hasattr(Image, "Resampling")
                    else Image.LANCZOS
                )

                if reduction_factor >= 3 and hasattr(image_util, "dpid_resize"):
                    self._log(f"DPID chosen for {file.name} (reduction factor: {reduction_factor:.2f})")
                    resized_img = image_util.dpid_resize(img, (new_width, new_height))
                else:
                    self._log(f"Using PIL LANCZOS resize for {file.name} (reduction factor: {reduction_factor:.2f})")
                    resized_img = img.resize((new_width, new_height), resample=resample_filter)

                save_kwargs = {}
                if file.suffix.lower() in [".jpg", ".jpeg"]:
                    save_kwargs["quality"] = 95
                    if "icc_profile" in img.info:
                        save_kwargs["icc_profile"] = img.info["icc_profile"]
                    if "exif" in img.info:
                        save_kwargs["exif"] = img.info["exif"]
                elif file.suffix.lower() == ".png":
                    save_kwargs["compress_level"] = 4

                if resized_img.mode == "P":
                    resized_img = resized_img.convert("RGB")

                resized_img.save(str(file), **save_kwargs)
            return "resized", None

        def summary(results: dict, extra_data: dict) -> str:
            return f"Completed resizing: {results.get('resized', 0)} resized, {results.get('skipped', 0)} skipped, {results.get('errors', 0)} errors"

        self._run_parallel_task(
            "Image Resizing",
            files,
            worker,
            summary,
            progress_callback,
            filter_func=self._filter_images_and_skip_masks
        )

    def process_alpha_images(
        self,
        files: list[Path],
        progress_callback: Callable[[float, str | None], None] | None = None,
    ) -> None:
        """Replace image transparency with a solid background color using parallel processing."""
        import re

        self._log("Processing transparent images (excluding mask files)...")
        cfg = self.config_data
        bg_color_str = cfg["alpha_bg_color"].strip()
        use_white_fallback = False

        try:
            if bg_color_str.lower() in ("-1", "random"):
                r, g, b = (random.randint(0, 255) for _ in range(3))
                bg_color_tuple = (r, g, b)
                self._log(f"Using random background color: #{r:02x}{g:02x}{b:02x}")
            else:
                if bg_color_str.startswith("#") and not re.fullmatch(
                    r"#([0-9a-fA-F]{3}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})", bg_color_str
                ):
                    raise ValueError("Invalid hex color format")

                color = ImageColor.getrgb(bg_color_str)
                # Ensure we have a 3-channel RGB tuple for the background
                bg_color_tuple = color[:3]
            self._log(f"Using background color: {bg_color_str} (RGB: {bg_color_tuple})")
        except (ValueError, TypeError) as e:
            self._log(f"Invalid color '{bg_color_str}': {e}. Using white instead.")
            bg_color_tuple = (255, 255, 255)  # White
            use_white_fallback = True  # Set flag to use white fallback

        def worker(file: Path) -> tuple[str, object | None]:
            with Image.open(file) as img:
                if img.mode not in ("RGBA", "LA"):
                    return "skipped", None

                # Always use white if the fallback flag is set
                color_to_use = (255, 255, 255) if use_white_fallback else bg_color_tuple

                background = Image.new("RGB", img.size, color_to_use)
                background.paste(img, (0, 0), img)
                background.save(str(file))
                return "processed", None

        def summary(results: dict, extra_data: dict) -> str:
            return f"Transparency processing complete: {results.get('processed', 0)} processed, {results.get('skipped', 0)} skipped, {results.get('errors', 0)} errors"

        self._run_parallel_task(
            "Transparency Processing",
            files,
            worker,
            summary,
            progress_callback,
            filter_func=self._filter_images_and_skip_masks,
        )

    def optimize_pngs(
        self,
        files: list[Path],
        progress_callback: Callable[[float, str | None], None] | None = None,
    ) -> None:
        """Optimize PNG files using oxipng in parallel."""
        self._log("Starting PNG optimization...")

        def worker(file: Path) -> tuple[str, object | None]:
            original_size = file.stat().st_size
            oxipng.optimize(file, level=5, fix_errors=True)
            new_size = file.stat().st_size
            bytes_saved = original_size - new_size
            if bytes_saved > 0:
                return "optimized", bytes_saved
            else:
                return "skipped", None

        def summary(results: dict, extra_data: dict) -> str:
            optimized_count = results.get("optimized", 0)
            total_bytes_saved = sum(extra_data.get("optimized", []))

            msg = f"Completed optimization: {optimized_count} PNGs optimized, {results.get('skipped', 0)} skipped, {results.get('errors', 0)} errors."
            if optimized_count > 0:
                avg_saving = total_bytes_saved / optimized_count
                msg += f"\nTotal bytes saved: {total_bytes_saved:,} (avg: {avg_saving:,.1f} per file)"
            return msg

        self._run_parallel_task(
            task_name="Optimize PNGs",
            files=files,
            worker_func=worker,
            summary_func=summary,
            progress_callback=progress_callback,
            filter_func=lambda f: f.suffix.lower() == ".png",
        )


class ProgressThrottler:
    """Helper class to throttle progress updates to the UI."""

    __slots__ = (
        "update_func",
        "min_interval",
        "last_update_time",
        "last_progress",
        "last_message",
        "lock",
    )

    def __init__(
        self,
        update_func: Callable[[float, str], None],
        min_interval: float = 0.1,
    ) -> None:
        self.update_func = update_func
        self.min_interval = min_interval
        self.last_update_time = 0.0
        self.last_progress = 0.0
        self.last_message = ""
        self.lock = threading.Lock()

    def update(self, progress: float, message: str | None = None) -> bool:
        current_time = time.time()
        with self.lock:
            force_update = (
                message is not None
                and message != self.last_message
                or abs(progress - self.last_progress) >= 0.01
            )
            if (
                force_update
                or (current_time - self.last_update_time)
                >= self.min_interval
            ):
                self.last_update_time = current_time
                self.last_progress = progress
                if message is not None:
                    self.last_message = message
                return True
        return False

    def __call__(
        self, progress: float, message: str | None = None
    ) -> None:
        if self.update(progress, message):
            self.update_func(progress, message or self.last_message)


class FileOperationsWindow(ctk.CTkToplevel):
    WINDOW_WIDTH: int = 440
    WINDOW_HEIGHT: int = 690
    MAX_WORKERS: int = max(
        4, os.cpu_count() or 4
    )  # Use at least 4 workers

    def __init__(
        self,
        parent: ctk.CTk,
        initial_dir: str | None = None,
        *args,
        **kwargs,
    ) -> None:
        """Initialize the FileOperationsWindow."""
        super().__init__(parent, *args, **kwargs)
        self.parent = parent
        self.dir_path = initial_dir or ""
        self.config_data = {
            "verify_images": False,
            "sequential_rename": False,
            "process_alpha": False,
            "alpha_bg_color": "#FFFFFF",
            "resize_large_images": False,
            "resize_megapixels": "Compute Proof (4MP)",
            "resize_custom_megapixels": 4,
            "optimization_type": "None",
        }
        self.config_state = UIState(self, self.config_data)
        self.executor = ThreadPoolExecutor(max_workers=self.MAX_WORKERS)
        self.message_queue: Queue = Queue()
        self.file_processor = FileProcessor(
            self.config_data,
            self.message_queue,
            lambda: self.cancel_requested,
            self.MAX_WORKERS,
        )
        self.processing_active: bool = False
        self.cancel_requested: bool = False

        self._setup_window()
        self._create_layout()

        if initial_dir:
            self.dir_path_var.set(initial_dir)

        # Start processing UI messages from worker threads

        self.after(100, self._process_message_queue)

    def _setup_window(self) -> None:
        """Configure window properties and initialize variables."""
        self.title("File Operations")
        self.geometry(f"{self.WINDOW_WIDTH}x{self.WINDOW_HEIGHT}")
        self.resizable(True, True)
        self.transient(self.parent)
        self.attributes("-topmost", True)
        self.wait_visibility()
        self.grab_set()
        self.focus_force()
        self.lift()
        self.after(300, self._ensure_focus)

        self.dir_path_var = ctk.StringVar(value=self.dir_path)
        self.status_var = ctk.StringVar(value="Ready")
        self.progress_var = ctk.DoubleVar(value=0)

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.wait_visibility()
        self.after(200, lambda: set_window_icon(self))

    def _on_close(self) -> None:
        """Handle window close: cancel ongoing operations and clean up."""
        if self.processing_active:
            self.cancel_requested = True
            self._log("Cancelling operations, please wait...")
            self.after(100, self._check_cancel_complete)
        else:
            self._cleanup()
            self.destroy()

    def _check_cancel_complete(self) -> None:
        """Check if cancel completed, then destroy window."""
        if not self.processing_active:
            self._cleanup()
            self.destroy()
        else:
            self.after(100, self._check_cancel_complete)

    def _cleanup(self) -> None:
        """Clean up resources before destroying window."""
        try:
            self.executor.shutdown(wait=False)
        except Exception as e:
            logger.error(f"Error shutting down thread pool: {e}")

    def _ensure_focus(self) -> None:
        """Ensure the window remains focused."""
        self.attributes("-topmost", False)
        self.focus_force()
        self.lift()

    def _validate_custom_mp(self, value_if_allowed: str) -> bool:
        """Validate the custom megapixel entry."""
        return value_if_allowed == "" or (
            value_if_allowed.isdigit()
            and len(value_if_allowed) <= 2
            and int(value_if_allowed) > 0
        )

    def _add_dropdown_tooltip(
        self, widget: tk.Widget, tooltip_text: str
    ) -> object:
        """Attach a tooltip to a widget (or update it if already present)."""
        from modules.util.ui.ToolTip import ToolTip

        if (
            hasattr(self, "_tooltip_registry")
            and widget in self._tooltip_registry
        ):
            old_tooltip = self._tooltip_registry[widget]
            if (
                hasattr(self, "_tooltip_objects")
                and old_tooltip in self._tooltip_objects
            ):
                self._tooltip_objects.remove(old_tooltip)
            widget.unbind("<Enter>")
            widget.unbind("<Leave>")
            widget.unbind("<ButtonPress>")
        tooltip = ToolTip(widget, text=tooltip_text, wide=True)
        self._tooltip_objects = getattr(self, "_tooltip_objects", [])
        self._tooltip_registry = getattr(self, "_tooltip_registry", {})
        self._tooltip_objects.append(tooltip)
        self._tooltip_registry[widget] = tooltip
        return tooltip

    def _update_optimization_tooltip(
        self, value: str, dropdown: tk.Widget, tooltips: dict
    ) -> None:
        """Update dropdown tooltip based on the selected optimization option."""
        tooltip_text = tooltips.get(
            value, "Select the type of image optimization to apply"
        )
        for child in dropdown.winfo_children():
            if isinstance(child, ctk.CTkToplevel):
                child.destroy()
        self._add_dropdown_tooltip(dropdown, tooltip_text)

    def _toggle_resize_options(self) -> None:
        """Enable/disable resize options based on the checkbox."""
        is_enabled = self.config_state.get_var("resize_large_images").get()
        self.resize_dropdown.configure(
            state="normal" if is_enabled else "disabled"
        )
        self._toggle_custom_mp_entry(self.resize_options_var.get())

    def _toggle_custom_mp_entry(self, selection: str) -> None:
        """Show/hide the custom megapixel entry based on dropdown selection."""
        is_enabled = self.config_state.get_var("resize_large_images").get()
        if is_enabled and selection == "Custom":
            self.custom_mp_entry.grid(
                row=0, column=2, padx=(2, 5), pady=2, sticky="w"
            )
        else:
            self.custom_mp_entry.grid_forget()

    def _create_layout(self) -> None:
        """Build the window layout."""
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)
        self._create_directory_frame()
        self._create_options_frame()
        self._create_status_frame()
        self._create_action_frame()

    def _create_directory_frame(self) -> None:
        """Create the directory selection UI."""
        dir_frame = ctk.CTkFrame(self)
        dir_frame.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="ew")
        dir_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(dir_frame, text="Directory:", anchor="w").grid(
            row=0, column=0, padx=5, pady=5, sticky="w"
        )
        path_frame = ctk.CTkFrame(dir_frame)
        path_frame.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        path_frame.grid_columnconfigure(0, weight=1)
        self.path_entry = ctk.CTkEntry(
            path_frame, textvariable=self.dir_path_var
        )
        self.path_entry.grid(
            row=0, column=0, padx=(5, 2), pady=5, sticky="ew"
        )
        ctk.CTkButton(
            path_frame,
            text="Browse...",
            width=100,
            command=self._browse_directory,
        ).grid(row=0, column=1, padx=(2, 5), pady=5)

    def _create_options_frame(self) -> None:
        """Create options (checkboxes and dropdowns) for file operations."""
        options_frame = ctk.CTkFrame(self)
        options_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        options_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            options_frame,
            text="File Operations:",
            anchor="w",
            font=("Segoe UI", 12, "bold"),
        ).grid(row=0, column=0, padx=5, pady=5, sticky="w")
        operations = [
            (
                "Verify Images for Corruption",
                "verify_images",
                "Check all images for corruption or format errors",
            ),
            (
                "Sequential Renaming (1.txt, 2.txt, etc.)",
                "sequential_rename",
                "Rename all files sequentially by file type",
            ),
            (
                "Replace Transperancy with Color",
                "process_alpha",
                "Replace transparent areas with a solid background color",
            ),
        ]
        for i, (text, key, tooltip) in enumerate(operations):
            checkbox = ctk.CTkCheckBox(
                options_frame,
                text=text,
                variable=self.config_state.get_var(key),
                onvalue=True,
                offvalue=False,
            )
            checkbox.grid(row=i + 1, column=0, padx=5, pady=2, sticky="w")
            if tooltip:
                self._add_dropdown_tooltip(checkbox, tooltip)

        # --- Resize Options ---
        resize_frame = ctk.CTkFrame(options_frame)
        resize_frame.grid(
            row=len(operations) + 1,
            column=0,
            padx=10,
            pady=(2, 5),
            sticky="w",
        )

        resize_checkbox = ctk.CTkCheckBox(
            resize_frame,
            text="Resize Images Above:",
            variable=self.config_state.get_var("resize_large_images"),
            onvalue=True,
            offvalue=False,
            command=self._toggle_resize_options,
        )
        resize_checkbox.grid(row=0, column=0, padx=(0, 5), pady=2)
        self._add_dropdown_tooltip(
            resize_checkbox,
            "Enable to resize images exceeding a specified megapixel count",
        )

        self.resize_options_var = self.config_state.get_var(
            "resize_megapixels"
        )
        self.resize_dropdown = ctk.CTkOptionMenu(
            resize_frame,
            variable=self.resize_options_var,
            values=[
                "1MP",
                "Compute Proof (4MP)",
                "Middleground (8MP)",
                "Zoom-in proof(16MP)",
                "Custom",
            ],
            command=self._toggle_custom_mp_entry,
            width=180,
        )
        self.resize_dropdown.grid(row=0, column=1, padx=0, pady=2)
        self._add_dropdown_tooltip(
            self.resize_dropdown,
            "Select the megapixel threshold for resizing",
        )

        vcmd = (self.register(self._validate_custom_mp), "%P")
        self.custom_mp_entry = ctk.CTkEntry(
            resize_frame,
            textvariable=self.config_state.get_var(
                "resize_custom_megapixels"
            ),
            width=40,
            placeholder_text="MP",
            validate="key",
            validatecommand=vcmd,
        )
        self._add_dropdown_tooltip(
            self.custom_mp_entry, "Enter custom megapixel limit (e.g., 12)"
        )

        alpha_row = len(operations) + 2
        color_frame = ctk.CTkFrame(options_frame)
        color_frame.grid(
            row=alpha_row, column=0, padx=5, pady=(2, 5), sticky="w"
        )
        ctk.CTkLabel(
            color_frame, text="Alpha Background Color:", anchor="w"
        ).grid(row=0, column=0, padx=5, pady=2, sticky="w")
        color_entry = ctk.CTkEntry(
            color_frame,
            width=100,
            textvariable=self.config_state.get_var("alpha_bg_color"),
            placeholder_text="#FFFFFF",
        )
        color_entry.grid(row=0, column=1, padx=5, pady=2, sticky="w")
        self._add_dropdown_tooltip(
            color_entry,
            "Enter color name (e.g., 'white', 'black'), hex code (e.g., '#FFFFFF'), or 'random'/-1 for random color",
        )

        current_row = alpha_row + 1
        opt_label = ctk.CTkLabel(
            options_frame, text="Image Optimization Type:", anchor="w"
        )
        opt_label.grid(
            row=current_row, column=0, padx=5, pady=(10, 2), sticky="w"
        )
        optimization_options = [
            "None",
            "Optimize PNGs",
            "Convert to WebP",
            "Convert to JPEG XL",
        ]
        opt_tooltips = {
            "None": "No image optimization will be applied",
            "Optimize PNGs": "Optimize PNGs using PyOxiPNG (level 5, fix_errors=True)",
            "Convert to WebP": "Re-encode all images to WebP format at 90% quality",
            "Convert to JPEG XL": "Encode images as JPEG XL at 90% quality or losslessly for JPEGs",
        }
        opt_var = self.config_state.get_var("optimization_type")
        opt_dropdown = ctk.CTkOptionMenu(
            options_frame,
            variable=opt_var,
            values=optimization_options,
            dynamic_resizing=True,
            width=200,
            command=lambda value: self._update_optimization_tooltip(
                value, opt_dropdown, opt_tooltips
            ),
        )
        opt_dropdown.grid(
            row=current_row + 1, column=0, padx=5, pady=(0, 5), sticky="w"
        )
        self._add_dropdown_tooltip(
            opt_dropdown,
            opt_tooltips.get(
                opt_var.get(),
                "Select the type of image optimization to apply",
            ),
        )
        self._add_dropdown_tooltip(
            opt_label, "Choose one optimization method for your images"
        )
        self.after(
            100,
            lambda: self._update_optimization_tooltip(
                opt_var.get(), opt_dropdown, opt_tooltips
            ),
        )
        self.after(100, self._toggle_resize_options)

    def _create_status_frame(self) -> None:
        """Create the status and progress UI."""
        status_frame = ctk.CTkFrame(self)
        status_frame.grid(row=2, column=0, padx=10, pady=5, sticky="nsew")
        status_frame.grid_columnconfigure(0, weight=1)
        status_frame.grid_rowconfigure(1, weight=1)
        ctk.CTkLabel(
            status_frame, textvariable=self.status_var, anchor="w"
        ).grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        self.progress_bar = ctk.CTkProgressBar(status_frame)
        self.progress_bar.grid(
            row=1, column=0, padx=5, pady=5, sticky="ew"
        )
        self.progress_bar.set(0)
        self.log_text = ctk.CTkTextbox(status_frame, height=150)
        self.log_text.grid(row=2, column=0, padx=5, pady=5, sticky="nsew")
        self.log_text.configure(state="disabled")

    def _create_action_frame(self) -> None:
        """Create the action buttons."""
        action_frame = ctk.CTkFrame(self)
        action_frame.grid(
            row=3, column=0, padx=10, pady=(5, 10), sticky="ew"
        )
        action_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkButton(
            action_frame,
            text="Process Files",
            command=self._process_files,
            fg_color="#28a745",
            hover_color="#218838",
        ).grid(row=0, column=0, padx=(5, 2), pady=10)
        ctk.CTkButton(
            action_frame, text="Close", command=self._on_close
        ).grid(row=0, column=2, padx=(2, 5), pady=10)

    def _browse_directory(self) -> None:
        """Open a directory selection dialog."""
        directory = filedialog.askdirectory(
            initialdir=self.dir_path_var.get() or os.path.expanduser("~")
        )
        if directory:
            self.dir_path_var.set(directory)
            self.dir_path = directory
            self._log(f"Directory selected: {directory}")

    def _log(self, message: str) -> None:
        """Queue a log message to be processed on the main thread."""
        self.message_queue.put(("log", message))

    def _update_status(
        self, message: str, progress: float | None = None
    ) -> None:
        """Queue a status update to be processed on the main thread."""
        self.message_queue.put(("status", message, progress))

    def _process_message_queue(self) -> None:
        """Process messages from worker threads."""
        try:
            if not self.winfo_exists():
                return
            while not self.message_queue.empty():
                action, *args = self.message_queue.get_nowait()
                if action == "log":
                    self._log_main_thread(args[0])
                elif action == "status":
                    self._update_status_main_thread(
                        args[0], args[1] if len(args) > 1 else None
                    )
                elif action == "operation_complete":
                    self._log_main_thread(f"Completed: {args[0]}")
                self.message_queue.task_done()
        except tk.TclError:
            return
        except Exception as e:
            logger.error(f"Error processing message queue: {e}")
        finally:
            with contextlib.suppress(tk.TclError):
                self.after(50, self._process_message_queue)

    def _log_main_thread(self, message: str) -> None:
        """Thread-safe log update on the main thread."""
        try:
            self.log_text.configure(state="normal")
            self.log_text.insert("end", message + "\n")
            self.log_text.see("end")
            self.log_text.configure(state="disabled")
            logger.info(message)
        except tk.TclError:
            pass

    def _update_status_main_thread(
        self, message: str, progress: float | None = None
    ) -> None:
        """Thread-safe status update on the main thread."""
        try:
            self.status_var.set(message)
            if progress is not None:
                self.progress_bar.set(progress)
            self.update_idletasks()
        except tk.TclError:
            pass

    def _update_progress(
        self,
        idx: int,
        total: int,
        message: str,
        progress_callback: Callable[[float, str | None], None]
        | None = None,
    ) -> None:
        """Helper to update progress."""
        progress = (idx + 1) / total
        if progress_callback:
            progress_callback(progress)
        else:
            self._update_status(message, progress)

    def _process_files(self) -> None:
        """Determine which file operations are selected and execute them in order."""
        directory = self.dir_path_var.get()
        if not directory:
            self._log("Error: No directory selected")
            return
        if not os.path.isdir(directory):
            self._log(f"Error: {directory} is not a valid directory")
            return
        try:
            path = Path(directory)
            files = [f for f in path.iterdir() if f.is_file()]
            self._log(f"Found {len(files)} files in {directory}")
        except Exception as e:
            self._log(f"Error scanning directory: {e}")
            return
        if not (
            self.config_data["verify_images"]
            or self.config_data["sequential_rename"]
            or self.config_data["process_alpha"]
            or self.config_data["resize_large_images"]
            or self.config_data["optimization_type"] != "None"
        ):
            self._log("No operations selected")
            return
        self._disable_ui()
        self.processing_active = True
        self.cancel_requested = False
        self.executor.submit(self._run_operations, files)

    def _disable_ui(self) -> None:
        """Disable UI elements during processing."""
        for widget in self.winfo_children():
            if isinstance(widget, ctk.CTkFrame):
                for child in widget.winfo_children():
                    if isinstance(
                        child,
                        ctk.CTkButton
                        | ctk.CTkCheckBox
                        | ctk.CTkOptionMenu,
                    ):
                        child.configure(state="disabled")

    def _enable_ui(self) -> None:
        """Re-enable UI elements after processing."""
        for widget in self.winfo_children():
            if isinstance(widget, ctk.CTkFrame):
                for child in widget.winfo_children():
                    if isinstance(
                        child,
                        ctk.CTkButton
                        | ctk.CTkCheckBox
                        | ctk.CTkOptionMenu,
                    ):
                        child.configure(state="normal")

    def _run_operations(self, files: list[Path]) -> None:
        """Run all selected operations in a background thread."""
        try:
            operations: list[
                tuple[
                    str,
                    Callable[
                        [
                            list[Path],
                            Callable[[float, str | None], None] | None,
                        ],
                        list[Path] | None,
                    ],
                ]
            ] = []
            if self.config_data["verify_images"]:
                operations.append(
                    (
                        "Image verification",
                        self.file_processor.verify_images,
                    )
                )
            if self.config_data["sequential_rename"]:
                operations.append(
                    (
                        "Sequential renaming",
                        self.file_processor.rename_files_sequentially,
                    )
                )
            if self.config_data["process_alpha"]:
                operations.append(
                    (
                        "Processing transparent images",
                        self.file_processor.process_alpha_images,
                    )
                )
            if self.config_data["resize_large_images"]:
                operations.append(
                    (
                        "Resizing large images",
                        self.file_processor.resize_large_images,
                    )
                )
            opt = self.config_data["optimization_type"]
            if opt == "Optimize PNGs":
                operations.append(
                    ("Optimizing PNGs", self.file_processor.optimize_pngs)
                )
            elif opt == "Convert to WebP":
                operations.append(
                    (
                        "Converting to WebP",
                        self.file_processor.convert_to_webp,
                    )
                )
            elif opt == "Convert to JPEG XL":
                operations.append(
                    (
                        "Converting to JPEG XL",
                        self.file_processor.convert_to_jpegxl,
                    )
                )
            operations_performed = False
            total_ops = len(operations)
            op_weight = 1.0 / total_ops if total_ops > 0 else 1.0
            for i, (name, op) in enumerate(operations):
                if self.cancel_requested:
                    self._log(f"Operation '{name}' cancelled")
                    break
                base_progress = i * op_weight

                def update_progress_fn(
                    step_progress: float,
                    msg: str | None = None,
                    _base_progress=base_progress,
                    _name=name,
                ) -> None:
                    overall = _base_progress + (step_progress * op_weight)
                    msg = (
                        msg
                        or f"Processing: {_name}... ({int(step_progress * 100)}%)"
                    )
                    self._update_status(msg, overall)

                throttled_progress = ProgressThrottler(update_progress_fn)
                self._update_status(f"Starting: {name}...", base_progress)
                try:
                    if name == "Sequential renaming":
                        files = (
                            op(files, progress_callback=throttled_progress)
                            or files
                        )
                    else:
                        op(files, progress_callback=throttled_progress)
                    if not self.cancel_requested:
                        self.message_queue.put(
                            ("operation_complete", name)
                        )
                        operations_performed = True
                except Exception as e:
                    self._log(f"Error during {name.lower()}: {e}")
            if not self.cancel_requested:
                self._update_status("Processing complete", 1.0)
                if (
                    operations_performed
                    and hasattr(self.parent, "file_manager")
                    and hasattr(self.parent.file_manager, "load_directory")
                ):
                    try:
                        self._log(
                            "Reloading file list in parent window..."
                        )
                        self.after(
                            0, self.parent.file_manager.load_directory
                        )
                    except Exception as e:
                        self._log(
                            f"Note: Could not refresh parent window's file list: {e}"
                        )
            else:
                self._update_status("Processing cancelled", 0.0)
        finally:
            self.after(0, self._enable_ui)
            self.processing_active = False
