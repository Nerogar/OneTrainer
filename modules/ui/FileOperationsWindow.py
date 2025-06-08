"""
OneTrainer File Operations Window

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
import threading
import time
import tkinter as tk
from collections.abc import Callable  # Updated import
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue
from tkinter import filedialog

from modules.util import path_util
from modules.util.ui.UIState import UIState

import customtkinter as ctk
import imagesize
import oxipng
from PIL import Image

# Set up module logger
logger = logging.getLogger(__name__)
for handler in logging.root.handlers:
    handler.setFormatter(logging.Formatter('%(message)s'))

# --- Constants ---
# Progress increments for sequential renaming and other operations
PROGRESS_INITIAL: float = 0.05
PROGRESS_RENAME_TEMP_PHASE_START: float = 0.15
PROGRESS_RENAME_TEMP_PHASE_RANGE: float = 0.3
PROGRESS_RENAME_EXEC_PHASE_START: float = 0.45
PROGRESS_RENAME_EXEC_PHASE_RANGE: float = 0.1
PROGRESS_RENAME_FINAL_PHASE_START: float = 0.55
PROGRESS_RENAME_FINAL_PHASE_RANGE: float = 0.4
PROGRESS_RENAME_FINAL_EXEC_PHASE_START: float = 0.95
PROGRESS_RENAME_FINAL_EXEC_PHASE_RANGE: float = 0.05

# Megapixel threshold (in pixels) for resizing images
MEGAPIXEL_THRESHOLD: int = 4_194_304


class ProgressThrottler:
    """Helper class to throttle progress updates to the UI."""
    def __init__(self, update_func: Callable[[float, str], None], min_interval: float = 0.1) -> None:
        self.update_func = update_func
        self.min_interval = min_interval
        self.last_update_time = 0.0
        self.last_progress = 0.0
        self.last_message = ""
        self.lock = threading.Lock()

    def update(self, progress: float, message: str | None = None) -> bool:
        current_time = time.time()
        with self.lock:
            force_update = (message is not None and message != self.last_message or
                            abs(progress - self.last_progress) >= 0.01)
            if force_update or (current_time - self.last_update_time) >= self.min_interval:
                self.last_update_time = current_time
                self.last_progress = progress
                if message is not None:
                    self.last_message = message
                return True
        return False

    def __call__(self, progress: float, message: str | None = None) -> None:
        if self.update(progress, message):
            self.update_func(progress, message or self.last_message)


class FileOperationsWindow(ctk.CTkToplevel):
    WINDOW_WIDTH: int = 440
    WINDOW_HEIGHT: int = 600
    MAX_WORKERS: int = max(4, os.cpu_count() or 4)  # Use at least 4 workers

    def __init__(
        self,
        parent: ctk.CTk,
        initial_dir: str | None = None,
        *args,
        **kwargs
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
            "optimization_type": "None"
        }
        self.config_state = UIState(self, self.config_data)

        # Initialize thread pool and messaging
        self.executor = ThreadPoolExecutor(max_workers=self.MAX_WORKERS)
        self.message_queue: Queue = Queue()
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
        for delay in [500, 1000, 2000]:
            self.after(delay, self._check_and_restore_focus)

    def _check_and_restore_focus(self) -> None:
        """Restore focus if lost."""
        try:
            if not self.focus_displayof():
                self.lift()
                self.focus_force()
        except (tk.TclError, RuntimeError, AttributeError):
            pass

    def _add_dropdown_tooltip(self, widget: tk.Widget, tooltip_text: str) -> object:
        """Attach a tooltip to a widget (or update it if already present)."""
        from modules.util.ui.ToolTip import ToolTip

        if hasattr(self, "_tooltip_registry") and widget in self._tooltip_registry:
            old_tooltip = self._tooltip_registry[widget]
            if hasattr(self, "_tooltip_objects") and old_tooltip in self._tooltip_objects:
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

    def _update_optimization_tooltip(self, value: str, dropdown: tk.Widget, tooltips: dict) -> None:
        """Update dropdown tooltip based on the selected optimization option."""
        tooltip_text = tooltips.get(value, "Select the type of image optimization to apply")
        for child in dropdown.winfo_children():
            if isinstance(child, ctk.CTkToplevel):
                child.destroy()
        self._add_dropdown_tooltip(dropdown, tooltip_text)

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
        self.path_entry = ctk.CTkEntry(path_frame, textvariable=self.dir_path_var)
        self.path_entry.grid(row=0, column=0, padx=(5, 2), pady=5, sticky="ew")
        ctk.CTkButton(
            path_frame, text="Browse...", width=100, command=self._browse_directory
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
            font=("Segoe UI", 12, "bold")
        ).grid(row=0, column=0, padx=5, pady=5, sticky="w")
        operations = [
            ("Verify Images for Corruption", "verify_images",
             "Check all images for corruption or format errors"),
            ("Sequential Renaming (1.txt, 2.txt, etc.)", "sequential_rename",
             "Rename all files sequentially by file type"),
            ("Replace Transperancy with Color", "process_alpha",
             "Replace transparent areas with a solid background color"),
            ("Resize Images Above 4MP", "resize_large_images",
             "Optimally resize images larger than 4 megapixels")
        ]
        for i, (text, key, tooltip) in enumerate(operations):
            checkbox = ctk.CTkCheckBox(
                options_frame,
                text=text,
                variable=self.config_state.get_var(key),
                onvalue=True,
                offvalue=False
            )
            checkbox.grid(row=i + 1, column=0, padx=5, pady=2, sticky="w")
            if tooltip:
                self._add_dropdown_tooltip(checkbox, tooltip)

        alpha_row = len(operations) + 1
        color_frame = ctk.CTkFrame(options_frame)
        color_frame.grid(row=alpha_row, column=0, padx=5, pady=(2, 5), sticky="w")
        ctk.CTkLabel(
            color_frame, text="Alpha Background Color:", anchor="w"
        ).grid(row=0, column=0, padx=5, pady=2, sticky="w")
        color_entry = ctk.CTkEntry(
            color_frame,
            width=100,
            textvariable=self.config_state.get_var("alpha_bg_color"),
            placeholder_text="#FFFFFF"
        )
        color_entry.grid(row=0, column=1, padx=5, pady=2, sticky="w")
        self._add_dropdown_tooltip(
            color_entry,
            "Enter color name (e.g., 'white', 'black'), hex code (e.g., '#FFFFFF'), or 'random'/-1 for random color"
        )

        current_row = alpha_row + 1
        opt_label = ctk.CTkLabel(
            options_frame, text="Image Optimization Type:", anchor="w"
        )
        opt_label.grid(row=current_row, column=0, padx=5, pady=(10, 2), sticky="w")
        optimization_options = [
            "None",
            "Optimize PNGs",
            "Convert to WebP",
            "Convert to JPEG XL"
        ]
        opt_tooltips = {
            "None": "No image optimization will be applied",
            "Optimize PNGs": "Optimize PNGs using PyOxiPNG (level 5, fix_errors=True)",
            "Convert to WebP": "Re-encode all images to WebP format at 90% quality",
            "Convert to JPEG XL": "Encode images as JPEG XL at 90% quality or losslessly for JPEGs"
        }
        opt_var = self.config_state.get_var("optimization_type")
        opt_dropdown = ctk.CTkOptionMenu(
            options_frame,
            variable=opt_var,
            values=optimization_options,
            dynamic_resizing=True,
            width=200,
            command=lambda value: self._update_optimization_tooltip(value, opt_dropdown, opt_tooltips)
        )
        opt_dropdown.grid(row=current_row + 1, column=0, padx=5, pady=(0, 5), sticky="w")
        self._add_dropdown_tooltip(
            opt_dropdown,
            opt_tooltips.get(opt_var.get(), "Select the type of image optimization to apply")
        )
        self._add_dropdown_tooltip(opt_label, "Choose one optimization method for your images")
        self.after(100, lambda: self._update_optimization_tooltip(opt_var.get(), opt_dropdown, opt_tooltips))

    def _create_status_frame(self) -> None:
        """Create the status and progress UI."""
        status_frame = ctk.CTkFrame(self)
        status_frame.grid(row=2, column=0, padx=10, pady=5, sticky="nsew")
        status_frame.grid_columnconfigure(0, weight=1)
        status_frame.grid_rowconfigure(1, weight=1)
        ctk.CTkLabel(status_frame, textvariable=self.status_var, anchor="w"
                     ).grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        self.progress_bar = ctk.CTkProgressBar(status_frame)
        self.progress_bar.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        self.progress_bar.set(0)
        self.log_text = ctk.CTkTextbox(status_frame, height=150)
        self.log_text.grid(row=2, column=0, padx=5, pady=5, sticky="nsew")
        self.log_text.configure(state="disabled")

    def _create_action_frame(self) -> None:
        """Create the action buttons."""
        action_frame = ctk.CTkFrame(self)
        action_frame.grid(row=3, column=0, padx=10, pady=(5, 10), sticky="ew")
        action_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkButton(
            action_frame,
            text="Process Files",
            command=self._process_files,
            fg_color="#28a745",
            hover_color="#218838"
        ).grid(row=0, column=0, padx=(5, 2), pady=10)
        ctk.CTkButton(
            action_frame,
            text="Close",
            command=self.destroy
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

    def _update_status(self, message: str, progress: float | None = None) -> None:
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
                        self._update_status_main_thread(args[0], args[1] if len(args) > 1 else None)
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

    def _update_status_main_thread(self, message: str, progress: float | None = None) -> None:
        """Thread-safe status update on the main thread."""
        try:
            self.status_var.set(message)
            if progress is not None:
                self.progress_bar.set(progress)
            self.update_idletasks()
        except tk.TclError:
            pass

    def _update_progress(self, idx: int, total: int, message: str,
                         progress_callback: Callable[[float, str | None], None] | None = None) -> None:
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
        if not (self.config_data["verify_images"] or
                self.config_data["sequential_rename"] or
                self.config_data["process_alpha"] or
                self.config_data["resize_large_images"] or
                self.config_data["optimization_type"] != "None"):
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
                    if isinstance(child, ctk.CTkButton | ctk.CTkCheckBox| ctk.CTkOptionMenu):
                        child.configure(state="disabled")

    def _enable_ui(self) -> None:
        """Re-enable UI elements after processing."""
        for widget in self.winfo_children():
            if isinstance(widget, ctk.CTkFrame):
                for child in widget.winfo_children():
                    if isinstance(child, ctk.CTkButton | ctk.CTkCheckBox| ctk.CTkOptionMenu):
                        child.configure(state="normal")

    def _run_operations(self, files: list[Path]) -> None:
            """Run all selected operations in a background thread."""
            try:
                operations: list[tuple[str, Callable[[list[Path], Callable[[float, str | None], None] | None], list[Path] | None]]] = []
                if self.config_data["verify_images"]:
                    operations.append(("Image verification", self._verify_images))
                if self.config_data["sequential_rename"]:
                    operations.append(("Sequential renaming", self._rename_files_sequentially))
                if self.config_data["process_alpha"]:
                    operations.append(("Processing transparent images", self._process_alpha_images))
                if self.config_data["resize_large_images"]:
                    operations.append(("Resizing large images", self._resize_large_images))
                opt = self.config_data["optimization_type"]
                if opt == "Optimize PNGs":
                    operations.append(("Optimizing PNGs", self._optimize_pngs))
                elif opt == "Convert to WebP":
                    operations.append(("Converting to WebP", self._convert_to_webp))
                elif opt == "Convert to JPEG XL":
                    operations.append(("Converting to JPEG XL", self._convert_to_jpegxl))
                operations_performed = False
                total_ops = len(operations)
                op_weight = 1.0 / total_ops if total_ops > 0 else 1.0
                for i, (name, op) in enumerate(operations):
                    if self.cancel_requested:
                        self._log(f"Operation '{name}' cancelled")
                        break
                    base_progress = i * op_weight
                    def update_progress_fn(step_progress: float, msg: str | None = None, _base_progress=base_progress, _name=name) -> None:
                        overall = _base_progress + (step_progress * op_weight)
                        msg = msg or f"Processing: {_name}... ({int(step_progress * 100)}%)"
                        self._update_status(msg, overall)
                    throttled_progress = ProgressThrottler(update_progress_fn)
                    self._update_status(f"Starting: {name}...", base_progress)
                    try:
                        if name == "Sequential renaming":
                            files = op(files, progress_callback=throttled_progress) or files
                        else:
                            op(files, progress_callback=throttled_progress)
                        if not self.cancel_requested:
                            self.message_queue.put(("operation_complete", name))
                            operations_performed = True
                    except Exception as e:
                        self._log(f"Error during {name.lower()}: {e}")
                if not self.cancel_requested:
                    self._update_status("Processing complete", 1.0)
                    if operations_performed and hasattr(self.parent, 'file_manager') and hasattr(self.parent.file_manager, 'load_directory'):
                        try:
                            self._log("Reloading file list in parent window...")
                            self.after(0, self.parent.file_manager.load_directory)
                        except Exception as e:
                            self._log(f"Note: Could not refresh parent window's file list: {e}")
                else:
                    self._update_status("Processing cancelled", 0.0)
            finally:
                self.after(0, self._enable_ui)
                self.processing_active = False

    def _verify_single_image(self, file: Path) -> tuple[bool, str, bool]:
        """
        Verify a single image file for corruption.

        Note: The image is opened twice because Pillow's .verify() invalidates the image object.
        """
        try:
            with Image.open(file) as img:
                img.verify()
            with Image.open(file) as img:
                img.load()
                if hasattr(img, "getpixel"):
                    img.getpixel((0, 0))
            return True, "", False
        except Exception as e:
            return False, str(e), True

    def _verify_images(self, files: list[Path],
                       progress_callback: Callable[[float, str | None], None] | None = None) -> None:
        """Verify images for corruption using parallel processing."""
        self._log("Starting image verification process...")
        image_files = [f for f in files if path_util.is_supported_image_extension(f.suffix)]
        if not image_files:
            self._log("No image files found to verify")
            return
        self._log(f"Found {len(image_files)} images to verify")
        results = {"valid": 0, "corrupted": 0, "error_files": []}
        results_lock = threading.Lock()
        total_files = len(image_files)
        processed_count = [0]
        def process_image(idx_file: tuple[int, Path]) -> None:
            idx, file = idx_file
            if self.cancel_requested:
                return
            try:
                valid, error_msg, _ = self._verify_single_image(file)
                with results_lock:
                    processed_count[0] += 1
                    progress = processed_count[0] / total_files
                    if progress_callback:
                        progress_callback(progress, f"Verifying image: {file.name}")
                    if valid:
                        results["valid"] += 1
                        log_msg = f"✓ {file.name} is valid"
                    else:
                        results["corrupted"] += 1
                        results["error_files"].append((file.name, error_msg))
                        log_msg = f"✗ {file.name} is CORRUPTED: {error_msg}"
                    self._log(log_msg)
            except Exception as e:
                with results_lock:
                    processed_count[0] += 1
                    results["corrupted"] += 1
                    results["error_files"].append((file.name, str(e)))
                    self._log(f"! Error verifying {file.name}: {e}")
                    if progress_callback:
                        progress_callback(processed_count[0] / total_files)
        try:
            with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
                list(executor.map(process_image, enumerate(image_files)))
            if results["corrupted"] == 0:
                self._log(f"✓ Image verification complete: All {results['valid']} images are valid")
            else:
                self._log(f"⚠ Image verification complete: {results['valid']} valid, {results['corrupted']} corrupted/problematic")
                self._log("Problematic files:")
                error_types = {}
                for name, error in results["error_files"]:
                    self._log(f"  - {name}: {error}")
                    etype = error.split(":")[0] if ":" in error else error
                    error_types[etype] = error_types.get(etype, 0) + 1
                self._log("Error breakdown:")
                for etype, count in error_types.items():
                    self._log(f"  - {etype}: {count} files")
        except Exception as e:
            self._log(f"Error during parallel image verification: {e}")

    def _convert_image_format(self,
                            files: list[Path],
                            target_format: str,
                            skip_extensions: set,
                            format_options: dict,
                            progress_callback: Callable[[float, str | None], None] | None = None
                            ) -> None:
        """Generic image conversion function for multiple formats."""
        self._log(f"Starting conversion to {target_format} format...")
        format_ext = format_options.get('format_ext', '.unknown')
        pil_format = format_options.get('pil_format', '')
        lossless_extensions = format_options.get('lossless_extensions', set())
        is_lossless_check = format_options.get('is_lossless_check', lambda file, img: False)
        quality = format_options.get('quality', 90)

        # Filter files to exclude masks (identified by '-' in filename)
        image_files = [
            f for f in files
            if path_util.is_supported_image_extension(f.suffix) and
            f.suffix.lower() not in skip_extensions and
            '-' not in f.stem  # Skip mask files
        ]

        if not image_files:
            self._log(f"No suitable image files found to convert to {target_format}")
            return

        self._log(f"Found {len(image_files)} images to convert to {target_format}")
        results = {"success": 0, "skipped": 0, "errors": 0, "total_bytes_saved": 0}
        results_lock = threading.Lock()
        total_files = len(image_files)
        processed_count = [0]
        def process_image(idx_file: tuple[int, Path]) -> None:
            idx, file = idx_file
            if self.cancel_requested:
                return
            try:
                original_size = file.stat().st_size
                new_path = file.with_suffix(format_ext)
                with results_lock:
                    processed_count[0] += 1
                    if progress_callback:
                        progress_callback(processed_count[0] / total_files,
                                          f"Converting to {target_format}: {file.name}")
                with Image.open(file) as img:
                    is_lossless = file.suffix.lower() in lossless_extensions
                    is_lossless = is_lossless_check(file, img) or is_lossless
                    save_kwargs = {"quality": quality}
                    if is_lossless:
                        save_kwargs["lossless"] = True
                        self._log(f"Converting {file.name} to {target_format} using lossless encoding")
                    else:
                        self._log(f"Converting {file.name} to {target_format} using {quality}% quality")
                    save_kwargs.update(format_options.get('save_kwargs', {}))
                    img.save(new_path, pil_format, **save_kwargs)
                if new_path.exists():
                    new_size = new_path.stat().st_size
                    bytes_saved = original_size - new_size
                    percent_saved = (bytes_saved / original_size) * 100 if original_size > 0 else 0
                    if new_size < original_size:
                        file.unlink()
                        self._log(f"Converted {file.name} to {target_format}: "
                                  f"{bytes_saved:,} bytes saved ({percent_saved:.1f}%)")
                        with results_lock:
                            results["success"] += 1
                            results["total_bytes_saved"] += bytes_saved
                    else:
                        new_path.unlink()
                        self._log(f"Skipped {file.name}: {target_format} version would be "
                                  f"larger by {-bytes_saved:,} bytes")
                        with results_lock:
                            results["skipped"] += 1
            except Exception as e:
                self._log(f"Error converting {file.name} to {target_format}: {e}")
                if 'new_path' in locals() and new_path.exists():
                    with contextlib.suppress(FileNotFoundError, PermissionError):
                        new_path.unlink()
                with results_lock:
                    results["errors"] += 1
                    processed_count[0] += 1
                    if progress_callback:
                        progress_callback(processed_count[0] / total_files)
        try:
            with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
                list(executor.map(process_image, enumerate(image_files)))
            if results["success"] > 0:
                avg_saving = results["total_bytes_saved"] / results["success"]
                self._log(f"Completed {target_format} conversion: "
                          f"{results['success']} of {total_files} images converted")
                self._log(f"Total bytes saved: {results['total_bytes_saved']:,} "
                          f"(avg: {avg_saving:,.1f} per file)")
                if results["skipped"] > 0:
                    self._log(f"Skipped {results['skipped']} images where "
                              f"{target_format} would be larger")
            else:
                self._log(f"{target_format} conversion completed with 0 successful conversions")
            if results["errors"] > 0:
                self._log(f"Encountered {results['errors']} errors during conversion")
        except Exception as e:
            self._log(f"Error during parallel {target_format} conversion: {e}")

    def _convert_to_webp(self, files: list[Path],
                         progress_callback: Callable[[float, str | None], None] | None = None) -> None:
        """Convert images to WebP format using the generic converter."""
        format_options = {
            'format_ext': '.webp',
            'pil_format': 'WEBP',
            'lossless_extensions': {'.png', '.tiff', '.tif', '.bmp'},
            'is_lossless_check': lambda file, img: False,
            'quality': 90
        }
        skip_extensions = {'.webp', '.jxl', '.avif'}
        self._convert_image_format(
            files,
            "WebP",
            skip_extensions,
            format_options,
            progress_callback
        )

    def _convert_to_jpegxl(self, files: list[Path],
                           progress_callback: Callable[[float, str | None], None] | None = None) -> None:
        """Convert images to JPEG XL format using the generic converter."""
        format_options = {
            'format_ext': '.jxl',
            'pil_format': 'JXL',
            'lossless_extensions': set(),
            'is_lossless_check': lambda file, img: file.suffix.lower() in {".jpg", ".jpeg"},
            'quality': 90
        }
        skip_extensions = {'.jxl'}
        self._convert_image_format(
            files,
            "JPEG XL",
            skip_extensions,
            format_options,
            progress_callback
        )

    def _rename_files_sequentially(self, files: list[Path],
                                   progress_callback: Callable[[float, str | None], None] | None = None
                                   ) -> list[Path]:
            """Rename files sequentially using a transactional approach for error resilience."""
            import uuid
            self._log("Starting sequential file renaming...")

            # Group files into image files and their associated files (captions, masks)
            image_files = [f for f in files if path_util.is_supported_image_extension(f.suffix)]
            other_files = [f for f in files if not path_util.is_supported_image_extension(f.suffix)]

            # Create mappings for caption files and mask files
            file_groups = {}

            # Identify all caption files (same name as image but with .txt extension)
            for file in other_files:
                if file.suffix.lower() == '.txt':
                    # Find the corresponding image file (any supported extension)
                    for img_file in image_files:
                        if img_file.stem == file.stem:
                            if img_file not in file_groups:
                                file_groups[img_file] = {'caption': None, 'masks': []}
                            file_groups[img_file]['caption'] = file
                            break

            # Identify mask files (filename-masklabel.png format)
            for file in files:
                if file.stem.endswith("-masklabel"):
                    base_name = file.stem[:-len("-masklabel")]
                    for img_file in image_files:
                        if img_file.stem == base_name:
                            if img_file not in file_groups:
                                file_groups[img_file] = {'caption': None, 'masks': []}
                            file_groups[img_file]['masks'].append((file, "masklabel"))
                            break

            self._log(f"Found {len(image_files)} image files and {len(file_groups)} file groups")

            # Sort image files for sequential naming
            sorted_img_files = sorted(image_files, key=lambda x: x.name)
            total_files = len(sorted_img_files)

            if progress_callback:
                progress_callback(PROGRESS_INITIAL)

            # Check if renaming is actually needed
            needs_renaming = False
            used_indices = set()
            expected_index = 1
            for i, file in enumerate(sorted_img_files):
                if progress_callback and i % 10 == 0:
                    progress_callback(PROGRESS_INITIAL + (i / total_files * 0.1))
                if not file.stem.isdigit():
                    self._log(f"Non-numeric filename found: {file.name}. Renaming needed.")
                    needs_renaming = True
                    break
                current_index = int(file.stem)
                if current_index in used_indices or current_index != expected_index:
                    self._log(f"Issue with {file.name}: expected {expected_index}, got {current_index}. Renaming needed.")
                    needs_renaming = True
                    break
                used_indices.add(current_index)
                expected_index += 1

            if not needs_renaming:
                self._log("Files are already sequentially named. No renaming needed.")
                if progress_callback:
                    progress_callback(1.0)
                return sorted_img_files

            # Begin renaming process
            unique_id = uuid.uuid4().hex
            rename_map = collections.OrderedDict()
            final_files: list[Path] = []
            files_to_process = []  # Will contain all files that need renaming

            try:
                self._log("Creating temporary filenames to avoid conflicts...")

                # First create temporary names for all image files
                for i, file in enumerate(sorted_img_files):
                    if progress_callback:
                        progress_callback(PROGRESS_RENAME_TEMP_PHASE_START +
                                         (i / total_files * PROGRESS_RENAME_TEMP_PHASE_RANGE * 0.5))
                    ext = file.suffix.lower()
                    temp_name = file.parent / f"temp_{unique_id}_{i}{ext}"
                    rename_map[file] = temp_name
                    files_to_process.append(file)

                    # Add related files to the rename map
                    if file in file_groups:
                        group = file_groups[file]
                        # Caption file
                        if group['caption']:
                            caption_temp = file.parent / f"temp_{unique_id}_{i}.txt"
                            rename_map[group['caption']] = caption_temp
                            files_to_process.append(group['caption'])

                        # Mask files
                        for mask_file, mask_label in group['masks']:
                            mask_temp = file.parent / f"temp_{unique_id}_{i}-{mask_label}{mask_file.suffix}"
                            rename_map[mask_file] = mask_temp
                            files_to_process.append(mask_file)

                # Rename all files to temporary names
                temp_files = []
                for i, (source, target) in enumerate(rename_map.items()):
                    if progress_callback:
                        progress_callback(PROGRESS_RENAME_EXEC_PHASE_START +
                                        (i / len(files_to_process) * PROGRESS_RENAME_EXEC_PHASE_RANGE))
                    try:
                        source.rename(target)
                        temp_files.append(target)
                    except Exception as e:
                        self._log(f"Error in temporary rename {source.name} → {target.name}: {e}")

                        # Define safe rollback function to avoid try-except in loop
                        def safe_rollback(orig_file: Path, temp_file: Path) -> None:
                            """Safely roll back a rename operation."""
                            try:
                                if temp_file.exists():
                                    self._log(f"Rolling back: {temp_file.name} → {orig_file.name}")
                                    temp_file.rename(orig_file)
                            except Exception as rollback_error:
                                self._log(f"Error rolling back rename: {rollback_error}")

                        # Roll back all previous renames
                        for original, temp in list(rename_map.items())[:rename_map.items().index((source, target))]:
                            if temp in temp_files:  # Only rollback successful renames
                                safe_rollback(original, temp)
                                temp_files.remove(temp)

                        self._log("Sequential rename failed, original filenames restored")
                        if progress_callback:
                            progress_callback(1.0)
                        return sorted_img_files

                # Create mapping for final names
                self._log("Creating final sequential filenames...")
                final_rename_map = {}

                # Group temporary files by their base name (to keep track of related files)
                temp_file_groups = {}
                for temp_file in temp_files:
                    # Extract the index part from temp_{unique_id}_{index}
                    if "temp_" + unique_id + "_" in temp_file.stem:
                        parts = temp_file.stem.split("_")
                        if len(parts) >= 3:
                            index_part = parts[2]
                            # Handle mask files
                            if '-' in index_part:
                                index_part = index_part.split('-')[0]
                            if index_part.isdigit():
                                if index_part not in temp_file_groups:
                                    temp_file_groups[index_part] = []
                                temp_file_groups[index_part].append(temp_file)

                # Sort the indices
                sorted_indices = sorted(temp_file_groups.keys(), key=int)

                # Create final names based on sorted indices
                for i, idx in enumerate(sorted_indices):
                    group_files = temp_file_groups[idx]
                    seq_num = i + 1  # Start from 1

                    for temp_file in group_files:
                        # Determine if this is a main image, caption, or mask file
                        if '-' in temp_file.stem:
                            # This is a mask file
                            base, mask_label = temp_file.stem.rsplit('-', 1)
                            final_name = temp_file.parent / f"{seq_num}-{mask_label}{temp_file.suffix}"
                        elif temp_file.suffix.lower() == '.txt':
                            # This is a caption file
                            final_name = temp_file.parent / f"{seq_num}.txt"
                        else:
                            # This is a main image file
                            final_name = temp_file.parent / f"{seq_num}{temp_file.suffix}"

                        final_rename_map[temp_file] = final_name

                # Execute final renames
                for i, (source, target) in enumerate(final_rename_map.items()):
                    if progress_callback:
                        progress = PROGRESS_RENAME_FINAL_EXEC_PHASE_START + (i / len(temp_files) * PROGRESS_RENAME_FINAL_EXEC_PHASE_RANGE)
                        progress_callback(progress)

                    try:
                        source.rename(target)
                        # Only add the main image files to final_files
                        if path_util.is_supported_image_extension(target.suffix) and '-' not in target.stem:
                            final_files.append(target)
                        self._log(f"Final rename: {source.name} → {target.name}")
                    except Exception as e:
                        self._log(f"Error in final rename {source.name} → {target.name}: {e}")

                if progress_callback:
                    progress_callback(1.0)
                self._log(f"Sequential renaming complete: {len(final_files)} image files renamed with their associated files")
                return final_files

            except Exception as e:
                self._log(f"Unexpected error during sequential renaming: {e}")
                if progress_callback:
                    progress_callback(1.0)
                return sorted_img_files

    def _resize_large_images(self, files: list[Path],
                             progress_callback: Callable[[float, str | None], None] | None = None) -> None:
        """Resize images larger than the defined threshold using parallel processing."""
        self._log("Starting resizing of large images...")
        image_files = [f for f in files if path_util.is_supported_image_extension(f.suffix)]
        self._log(f"Found {len(image_files)} image files to check")
        results = {"resized": 0, "skipped": 0, "errors": 0}
        results_lock = threading.Lock()
        total_files = len(image_files)
        processed_count = [0]
        def process_image(idx_file: tuple[int, Path]) -> None:
            idx, file = idx_file
            if self.cancel_requested:
                return
            try:
                width, height = imagesize.get(file)
                total_pixels = width * height
                with results_lock:
                    processed_count[0] += 1
                    if progress_callback:
                        progress_callback(processed_count[0] / total_files, f"Analyzing image: {file.name}")
                if total_pixels > MEGAPIXEL_THRESHOLD:
                    scale_factor = (MEGAPIXEL_THRESHOLD / total_pixels) ** 0.5
                    new_width, new_height = int(width * scale_factor), int(height * scale_factor)
                    self._log(
                        f"Resizing {file.name} from {width}x{height} "
                        f"({total_pixels/1_000_000:.2f}MP) to {new_width}x{new_height} "
                        f"({MEGAPIXEL_THRESHOLD/1_000_000:.2f}MP)"
                    )
                    try:
                        with Image.open(file) as img:
                            resized_img = img.resize((new_width, new_height), Image.LANCZOS)
                            resized_img.save(file, quality=95)
                        with results_lock:
                            results["resized"] += 1
                    except Exception as e:
                        self._log(f"Error resizing {file.name}: {e}")
                        with results_lock:
                            results["errors"] += 1
                else:
                    with results_lock:
                        results["skipped"] += 1
            except Exception as e:
                self._log(f"Error getting dimensions for {file.name}: {e}")
                with results_lock:
                    results["errors"] += 1
                    processed_count[0] += 1
                    if progress_callback:
                        progress_callback(processed_count[0] / total_files)
        try:
            with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
                list(executor.map(process_image, enumerate(image_files)))
            self._log(f"Completed resizing: {results['resized']} images resized, {results['skipped']} skipped, {results['errors']} errors")
        except Exception as e:
            self._log(f"Error during parallel image resizing: {e}")

    def _process_alpha_images(self, files: list[Path],
                             progress_callback: Callable[[float, str | None], None] | None = None) -> None:
        """Replace image transparency with a solid background color using parallel processing."""
        # Filter to exclude mask files
        image_files = [
            f for f in files
            if path_util.is_supported_image_extension(f.suffix) and not f.stem.endswith("-masklabel")
        ]

        if not image_files:
            self._log("No image files found")
            return

        self._log("Processing transparent images (excluding mask files)...")
        bg_color = self.config_data["alpha_bg_color"].strip().lower()
        if bg_color in ("-1", "random"):
            import random
            r, g, b = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
            bg_color_tuple = (r, g, b)
            self._log(f"Using random background color: #{r:02x}{g:02x}{b:02x} (RGB: {bg_color_tuple})")
        else:
            try:
                if bg_color.startswith("#") and len(bg_color) == 7:
                    bg_color_tuple = tuple(int(bg_color[i:i+2], 16) for i in (1, 3, 5))
                else:
                    img = Image.new("RGB", (1, 1), bg_color)
                    bg_color_tuple = img.getpixel((0, 0))
                self._log(f"Using background color: {bg_color} (RGB: {bg_color_tuple})")
            except Exception as e:
                self._log(f"Invalid color '{bg_color}': {e}. Using white instead.")
                bg_color_tuple = (255, 255, 255)
        results = {"processed": 0, "skipped": 0, "errors": 0}
        results_lock = threading.Lock()
        total_files = len(image_files)
        processed_count = [0]
        def process_image(idx_file: tuple[int, Path]) -> None:
            idx, file = idx_file
            if self.cancel_requested:
                return
            try:
                with Image.open(file) as img:
                    if img.mode not in ("RGBA", "LA"):
                        with results_lock:
                            results["skipped"] += 1
                            processed_count[0] += 1
                            if progress_callback:
                                progress_callback(processed_count[0] / total_files, f"Processing: {file.name}")
                        return
                    has_transparency = (
                        any(a < 255 for a in img.getdata())
                        if img.mode == "LA"
                        else any(p[3] < 255 for p in img.getdata())
                    )
                    if not has_transparency:
                        with results_lock:
                            results["skipped"] += 1
                            processed_count[0] += 1
                            if progress_callback:
                                progress_callback(processed_count[0] / total_files)
                        return
                    background = Image.new("RGB", img.size, bg_color_tuple)
                    background.paste(img, (0, 0), img)
                    background.save(file)
                    with results_lock:
                        results["processed"] += 1
                        processed_count[0] += 1
                        if progress_callback:
                            progress_callback(processed_count[0] / total_files)
            except Exception as e:
                self._log(f"Error processing {file.name}: {e}")
                with results_lock:
                    results["errors"] += 1
                    processed_count[0] += 1
                    if progress_callback:
                        progress_callback(processed_count[0] / total_files)
        try:
            with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
                list(executor.map(process_image, enumerate(image_files)))
            self._log(f"Transparency processing complete: {results['processed']} processed, {results['skipped']} skipped, {results['errors']} errors")
        except Exception as e:
            self._log(f"Error during parallel transparency processing: {e}")

    def _optimize_pngs(self, files: list[Path],
                         progress_callback: Callable[[float, str | None], None] | None = None) -> None:
        """Optimize PNG files using oxipng."""
        self._log("Starting PNG optimization...")
        png_files = [f for f in files if f.suffix.lower() == ".png"]
        if not png_files:
            self._log("No PNG files found to optimize")
            return
        self._log(f"Found {len(png_files)} PNG files to optimize")
        success_count = 0
        total_bytes_saved = 0
        total_files = len(png_files)
        for idx, file in enumerate(png_files):
            self._update_progress(idx, total_files, f"Optimizing PNG: {file.name}", progress_callback)
            try:
                original_size = file.stat().st_size
            except Exception as e:
                self._log(f"Error getting original size for {file.name}: {e}")
                continue
            try:
                oxipng.optimize(file, level=5, fix_errors=True)
            except Exception as e:
                self._log(f"Error optimizing {file.name}: {e}")
                continue
            try:
                new_size = file.stat().st_size
                bytes_saved = original_size - new_size
                success_count += 1
                total_bytes_saved += bytes_saved
            except Exception as e:
                self._log(f"Error calculating results for {file.name}: {e}")
        if success_count:
            avg_saving = total_bytes_saved / success_count
            self._log(f"Completed optimization: {success_count} of {total_files} PNGs optimized")
            self._log(f"Total bytes saved: {total_bytes_saved:,} (avg: {avg_saving:,.1f} per file)")
        else:
            self._log("PNG optimization completed with 0 successful optimizations")
