"""
OneTrainer File Operations Window

This module defines a CTkToplevel window for performing common batch operations on files
such as renaming, resizing, and format conversion.
"""

import contextlib
import logging
import os
import tkinter as tk
from pathlib import Path
from tkinter import filedialog

from modules.util import path_util
from modules.util.ui.UIState import UIState

import customtkinter as ctk
import imagesize
import oxipng
import pillow_jxl  # noqa: F401  # Needed for plugin registration
from PIL import Image

# Set up module logger
logger = logging.getLogger(__name__)

# Configure logger to show only the message (no timestamp, filename, or line number)
for handler in logging.root.handlers:
    handler.setFormatter(logging.Formatter('%(message)s'))

class FileOperationsWindow(ctk.CTkToplevel):
    """Window for batch file operations such as renaming, resizing, and format conversion."""

    WINDOW_WIDTH: int = 440
    WINDOW_HEIGHT: int = 600

    def __init__(
        self,
        parent: ctk.CTk,
        initial_dir: str | None = None,
        *args,
        **kwargs
    ) -> None:
        """Initialize the FileOperationsWindow.

        Args:
            parent: The parent window
            initial_dir: Initial directory path
        """
        super().__init__(parent, *args, **kwargs)
        self.parent = parent
        self.dir_path = initial_dir or ""

        # UI state for checkboxes and dropdowns
        self.config_data = {
            "verify_images": False,
            "sequential_rename": False,
            "process_alpha": False,  # New option for alpha processing
            "alpha_bg_color": "#FFFFFF",  # Default background color
            "resize_large_images": False,
            "optimization_type": "None"
        }
        self.config_state = UIState(self, self.config_data)

        self._setup_window()
        self._create_layout()

        # If directory provided, update the path display
        if initial_dir:
            self.dir_path_var.set(initial_dir)

    def _setup_window(self) -> None:
        """Set up window properties."""
        self.title("File Operations")
        self.geometry(f"{self.WINDOW_WIDTH}x{self.WINDOW_HEIGHT}")
        self.resizable(True, True)

        # Make this window transient to the parent
        self.transient(self.parent)

        # Set this window to be on top until fully initialized
        self.attributes("-topmost", True)

        # Wait for window to be visible before setting focus
        self.wait_visibility()

        # Grab and force focus
        self.grab_set()
        self.focus_force()
        self.lift()

        # After a short delay, remove topmost but maintain focus
        self.after(300, lambda: self._ensure_focus())

        # Variables
        self.dir_path_var = ctk.StringVar(value=self.dir_path)
        self.status_var = ctk.StringVar(value="Ready")
        self.progress_var = ctk.DoubleVar(value=0)

    def _ensure_focus(self) -> None:
        """Ensure window stays focused and visible."""
        self.attributes("-topmost", False)  # Remove topmost attribute
        self.focus_force()
        self.lift()

        # Schedule periodic focus check for the first few seconds
        # This helps with scenarios where OS or other apps might steal focus
        for delay in [500, 1000, 2000]:
            self.after(delay, lambda: self._check_and_restore_focus())

    def _check_and_restore_focus(self) -> None:
        """Check if window has focus, restore if needed."""
        try:
            if not self.focus_displayof():
                self.lift()
                self.focus_force()
        except (tk.TclError, RuntimeError, AttributeError):
            # Ignore errors if window is destroyed
            pass

    def _add_dropdown_tooltip(self, dropdown_widget, tooltip_text):
        """Add a tooltip to a dropdown widget that properly handles hover events."""
        from modules.util.ui.ToolTip import ToolTip

        # First, remove any existing tooltip that might be attached to this widget
        if hasattr(self, '_tooltip_registry') and dropdown_widget in self._tooltip_registry:
            # Remove the old tooltip from our objects list
            old_tooltip = self._tooltip_registry[dropdown_widget]
            if hasattr(self, '_tooltip_objects') and old_tooltip in self._tooltip_objects:
                self._tooltip_objects.remove(old_tooltip)

            # Unbind tooltip events
            dropdown_widget.unbind("<Enter>")
            dropdown_widget.unbind("<Leave>")
            dropdown_widget.unbind("<ButtonPress>")

        # Create tooltip directly using the ToolTip class
        tooltip = ToolTip(dropdown_widget, text=tooltip_text, wide=True)

        # Store the tooltip objects by widget reference
        if not hasattr(self, '_tooltip_objects'):
            self._tooltip_objects = []
        if not hasattr(self, '_tooltip_registry'):
            self._tooltip_registry = {}

        self._tooltip_objects.append(tooltip)
        self._tooltip_registry[dropdown_widget] = tooltip

        return tooltip

    def _update_optimization_tooltip(self, value, dropdown, tooltips):
        """Update the tooltip for the dropdown based on the selected value."""
        tooltip_text = tooltips.get(value, "Select the type of image optimization to apply")

        # First remove any existing tooltip
        for child in dropdown.winfo_children():
            if isinstance(child, ctk.CTkToplevel):
                child.destroy()

        # Apply tooltip directly using our custom method
        self._add_dropdown_tooltip(dropdown, tooltip_text)

    def _create_layout(self) -> None:
        """Create the main UI layout."""
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)  # Status area can expand

        self._create_directory_frame()
        self._create_options_frame()
        self._create_status_frame()
        self._create_action_frame()

    def _create_directory_frame(self) -> None:
        """Create the directory selection area."""
        dir_frame = ctk.CTkFrame(self)
        dir_frame.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="ew")
        dir_frame.grid_columnconfigure(0, weight=1)

        # Directory path label
        ctk.CTkLabel(
            dir_frame,
            text="Directory:",
            anchor="w"
        ).grid(row=0, column=0, padx=5, pady=5, sticky="w")

        # Directory selection area
        path_frame = ctk.CTkFrame(dir_frame)
        path_frame.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        path_frame.grid_columnconfigure(0, weight=1)

        # Path entry and browse button
        self.path_entry = ctk.CTkEntry(
            path_frame,
            textvariable=self.dir_path_var
        )
        self.path_entry.grid(row=0, column=0, padx=(5, 2), pady=5, sticky="ew")

        ctk.CTkButton(
            path_frame,
            text="Browse...",
            width=100,
            command=self._browse_directory
        ).grid(row=0, column=1, padx=(2, 5), pady=5)

    def _create_options_frame(self) -> None:
        """Create checkboxes and dropdowns for file operations."""
        options_frame = ctk.CTkFrame(self)
        options_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        options_frame.grid_columnconfigure(0, weight=1)

        # Operations header
        ctk.CTkLabel(
            options_frame,
            text="File Operations:",
            anchor="w",
            font=("Segoe UI", 12, "bold")
        ).grid(row=0, column=0, padx=5, pady=5, sticky="w")

        # Checkboxes for operations
        operations = [
            ("Verify Images for Corruption", "verify_images",
             "Check all images for corruption or format errors"),

            ("Sequential Renaming (1.txt, 2.txt, etc.)", "sequential_rename",
            "Rename all files sequentially by file type"),

            ("Process Images with Transparency", "process_alpha",
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
                # Use our custom tooltip method instead of components.add_tooltip
                self._add_dropdown_tooltip(checkbox, tooltip)

        # Add color input for alpha processing
        alpha_row = len(operations) + 1
        color_frame = ctk.CTkFrame(options_frame)
        color_frame.grid(row=alpha_row, column=0, padx=5, pady=(2, 5), sticky="w")

        # Color input label and field
        ctk.CTkLabel(
            color_frame,
            text="Alpha Background Color:",
            anchor="w"
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

        # Add the optimization type dropdown
        current_row = alpha_row + 1

        # Label for optimization dropdown
        opt_label = ctk.CTkLabel(
            options_frame,
            text="Image Optimization Type:",
            anchor="w"
        )
        opt_label.grid(row=current_row, column=0, padx=5, pady=(10, 2), sticky="w")

        # Dropdown for optimization type
        optimization_options = [
            "None",
            "Optimize PNGs",
            "Convert to WebP",
            "Convert to JPEG XL"
        ]

        # Add tooltips for optimization types
        opt_tooltips = {
            "None": "No image optimization will be applied",
            "Optimize PNGs": "Optimize PNGs using PyOxiPNG (level 5, fix_errors=True)",
            "Convert to WebP": "Re-encode all images to WebP format at 90% quality",
            "Convert to JPEG XL": "Encode images as JPEG XL at 90% quality or losslessly for JPEGs"
        }

        # Create variable for the dropdown and set up callback for tooltip updates
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

        # Force initial tooltip update
        self.after(100, lambda: self._update_optimization_tooltip(opt_var.get(), opt_dropdown, opt_tooltips))

    def _create_status_frame(self) -> None:
        """Create area for status messages and progress bar."""
        status_frame = ctk.CTkFrame(self)
        status_frame.grid(row=2, column=0, padx=10, pady=5, sticky="nsew")
        status_frame.grid_columnconfigure(0, weight=1)
        status_frame.grid_rowconfigure(1, weight=1)

        # Status label
        ctk.CTkLabel(
            status_frame,
            textvariable=self.status_var,
            anchor="w"
        ).grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(status_frame)
        self.progress_bar.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        self.progress_bar.set(0)

        # Log output area
        self.log_text = ctk.CTkTextbox(status_frame, height=150)
        self.log_text.grid(row=2, column=0, padx=5, pady=5, sticky="nsew")
        self.log_text.configure(state="disabled")

    def _create_action_frame(self) -> None:
        """Create buttons for actions."""
        action_frame = ctk.CTkFrame(self)
        action_frame.grid(row=3, column=0, padx=10, pady=(5, 10), sticky="ew")
        action_frame.grid_columnconfigure(1, weight=1)

        # Process button
        ctk.CTkButton(
            action_frame,
            text="Process Files",
            command=self._process_files,
            fg_color="#28a745",
            hover_color="#218838"
        ).grid(row=0, column=0, padx=(5, 2), pady=10)

        # Close button
        ctk.CTkButton(
            action_frame,
            text="Close",
            command=self.destroy
        ).grid(row=0, column=2, padx=(2, 5), pady=10)

    def _browse_directory(self) -> None:
        """Open directory browser dialog."""
        directory = filedialog.askdirectory(
            initialdir=self.dir_path_var.get() or os.path.expanduser("~")
        )
        if directory:
            self.dir_path_var.set(directory)
            self.dir_path = directory
            self._log(f"Directory selected: {directory}")

    def _log(self, message: str) -> None:
        """Add message to the log text area."""
        self.log_text.configure(state="normal")
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")
        logger.info(message)

    def _update_status(self, message: str, progress: float = None) -> None:
        """Update status message and progress bar."""
        self.status_var.set(message)
        if progress is not None:
            self.progress_bar.set(progress)
        self.update()

    def _process_files(self) -> None:
            """Process files with selected operations."""
            parent_window = self.parent
            directory = self.dir_path_var.get()
            if not directory:
                self._log("Error: No directory selected")
                return

            if not os.path.isdir(directory):
                self._log(f"Error: {directory} is not a valid directory")
                return

            # Get list of files
            try:
                path = Path(directory)
                files = [f for f in path.iterdir() if f.is_file()]
                self._log(f"Found {len(files)} files in {directory}")
            except Exception as e:
                self._log(f"Error scanning directory: {e}")
                return

            # Check if any operation is selected
            if not (self.config_data["verify_images"] or
                    self.config_data["sequential_rename"] or
                    self.config_data["process_alpha"] or  # New option
                    self.config_data["resize_large_images"] or
                    self.config_data["optimization_type"] != "None"):
                self._log("No operations selected")
                return

            # Execute selected operations in the specified order
            operations = []

            # Run verification first if selected
            if self.config_data["verify_images"]:
                operations.append(("Image verification", self._verify_images))

            if self.config_data["sequential_rename"]:
                operations.append(("Sequential renaming", self._rename_files_sequentially))

            # Add new alpha channel processing step
            if self.config_data["process_alpha"]:
                operations.append(("Processing transparent images", self._process_alpha_images))

            if self.config_data["resize_large_images"]:
                operations.append(("Resizing large images", self._resize_large_images))

            # Add optimization operation based on dropdown selection
            optimization_type = self.config_data["optimization_type"]
            if optimization_type == "Optimize PNGs":
                operations.append(("Optimizing PNGs", self._optimize_pngs))
            elif optimization_type == "Convert to WebP":
                operations.append(("Converting to WebP", self._convert_to_webp))
            elif optimization_type == "Convert to JPEG XL":
                operations.append(("Converting to JPEG XL", self._convert_to_jpegxl))

            # Track if any operations were actually performed
            operations_performed = False

            # Setup progress tracking for overall operations
            total_operations = len(operations)
            operation_weight = 1.0 / total_operations if total_operations > 0 else 1.0

            # Process each operation
            for i, (name, operation) in enumerate(operations):
                base_progress = i * operation_weight

                # Define a progress callback for the current operation
                # Fix B023 by binding loop variables as default arguments
                def update_progress(step_progress, _base=base_progress, _name=name):
                    # Calculate overall progress: base + (current operation's progress * weight)
                    overall_progress = _base + (step_progress * operation_weight)
                    self._update_status(f"Processing: {_name}... ({int(step_progress * 100)}%)", overall_progress)

                self._update_status(f"Starting: {name}...", base_progress)
                try:
                    # If the operation is renaming, we need to update the files list
                    if name == "Sequential renaming":
                        files = operation(files, progress_callback=update_progress) or files
                    else:
                        operation(files, progress_callback=update_progress)
                    self._log(f"Completed: {name}")
                    operations_performed = True
                except Exception as e:
                    self._log(f"Error during {name.lower()}: {e}")

            self._update_status("Processing complete", 1.0)

            # If any operations were performed, refresh the parent window's file list
            if operations_performed and hasattr(parent_window, 'file_manager') and hasattr(parent_window.file_manager, 'load_directory'):
                    try:
                        self._log("Reloading file list in parent window...")
                        parent_window.file_manager.load_directory()

                        # Use robust focus handling
                        self.lift()
                        self.focus_force()

                        # Ensure window stays focused
                        self.after(100, self._ensure_focus)
                    except Exception as e:
                        self._log(f"Note: Could not refresh parent window's file list: {e}")

    def _process_alpha_images(self, files: list[Path], progress_callback=None) -> None:
        """Process images with alpha channels by replacing transparency with a solid color."""
        self._log("Processing transparent images...")
        bg_color = self.config_data["alpha_bg_color"].strip().lower()

        # Handle random color option
        if bg_color == "-1" or bg_color == "random":
            import random
            # Generate random RGB values
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            bg_color_tuple = (r, g, b)
            hex_color = f"#{r:02x}{g:02x}{b:02x}"
            self._log(f"Using random background color: {hex_color} (RGB: {bg_color_tuple})")
        else:
            # Validate color input
            try:
                if bg_color.startswith('#') and len(bg_color) == 7:  # #RRGGBB format
                    bg_color_tuple = tuple(int(bg_color[i:i+2], 16) for i in (1, 3, 5))
                else:
                    # Validate color name by creating a test image
                    img = Image.new("RGB", (1, 1), bg_color)
                    bg_color_tuple = img.getpixel((0, 0))
                self._log(f"Using background color: {bg_color} (RGB: {bg_color_tuple})")
            except Exception as e:
                self._log(f"Invalid color '{bg_color}': {e}. Using white instead.")
                bg_color_tuple = (255, 255, 255)

        # Find and process images
        image_files = [f for f in files if path_util.is_supported_image_extension(f.suffix)]
        if not image_files:
            self._log("No image files found")
            return

        processed, skipped = 0, 0
        total = len(image_files)

        for idx, file in enumerate(image_files):
            try:
                # Update progress
                if progress_callback:
                    progress_callback((idx + 1) / total)
                else:
                    self._update_status(f"Processing: {file.name}", (idx + 1) / total)

                # Process the image
                with Image.open(file) as img:
                    # Skip if not RGBA or LA mode
                    if img.mode not in ('RGBA', 'LA'):
                        skipped += 1
                        continue

                    # Check for actual transparency
                    has_transparency = False
                    if img.mode == 'RGBA':
                        has_transparency = any(a < 255 for _, _, _, a in img.getdata())
                    else:  # LA mode
                        has_transparency = any(a < 255 for _, a in img.getdata())

                    if not has_transparency:
                        skipped += 1
                        continue

                    # Replace transparency with background color
                    background = Image.new('RGB', img.size, bg_color_tuple)
                    background.paste(img, (0, 0), img)
                    background.save(file)
                    processed += 1

            except Exception as e:
                self._log(f"Error processing {file.name}: {e}")

        self._log(f"Transparency processing complete: {processed} processed, {skipped} skipped")

    def _verify_single_image(self, file: Path) -> tuple[bool, str, bool]:
        """Verify a single image file for corruption.

        Returns:
            tuple: (is_valid, error_message, is_format_error)
            is_format_error distinguishes between format-specific and general errors
        """
        try:
            # First try to open the image to detect basic format issues
            with Image.open(file) as img:
                # Verify checks structural integrity but doesn't decode fully
                img.verify()

            # If verify passed, also try to load a small part to detect pixel-level corruption
            # This is more thorough as verify() only checks headers and structure
            with Image.open(file) as img:
                # Load just a small part of the image to check data integrity
                # This forces PIL to actually decode some image data
                img.load()
                # Try to access pixel data from a small region
                if hasattr(img, 'getpixel'):
                    img.getpixel((0, 0))

            return True, "", False
        except Exception as e:
            return False, str(e), True

    def _verify_images(self, files: list[Path], progress_callback=None) -> None:
        """Verify images for corruption using PIL's verify method."""
        self._log("Starting image verification process...")

        # Find all image files
        image_files = [
            f for f in files
            if path_util.is_supported_image_extension(f.suffix)
        ]

        if not image_files:
            self._log("No image files found to verify")
            return

        self._log(f"Found {len(image_files)} images to verify")

        # Track verification statistics
        valid_count = 0
        corrupted_count = 0
        error_files = []

        # Process each image file
        total_files = len(image_files)
        for idx, file in enumerate(image_files):
            # Update progress
            if progress_callback:
                progress_callback((idx + 1) / total_files)
            else:
                progress = (idx + 1) / total_files
                self._update_status(f"Verifying image: {file.name}", progress)

            try:
                # Use helper function to verify the image
                is_valid, error_msg, is_format_error = self._verify_single_image(file)

                if is_valid:
                    self._log(f"✓ {file.name} is valid")
                    valid_count += 1
                else:
                    # Format-specific error from verification
                    self._log(f"✗ {file.name} is CORRUPTED: {error_msg}")
                    corrupted_count += 1
                    error_files.append((file.name, error_msg))

            except Exception as e:
                # General error (e.g. file not found)
                self._log(f"! Error verifying {file.name}: {e}")
                corrupted_count += 1
                error_files.append((file.name, str(e)))

        # Log summary
        if corrupted_count == 0:
            self._log(f"✓ Image verification complete: All {valid_count} images are valid")
        else:
            self._log(f"⚠ Image verification complete: {valid_count} valid, {corrupted_count} corrupted/problematic")
            self._log("Problematic files:")
            for name, error in error_files:
                self._log(f"  - {name}: {error}")

            # Group errors by type for better analysis
            error_types = {}
            for _, error in error_files:
                error_type = error.split(':')[0] if ':' in error else error
                if error_type not in error_types:
                    error_types[error_type] = 0
                error_types[error_type] += 1

            self._log("Error breakdown:")
            for error_type, count in error_types.items():
                self._log(f"  - {error_type}: {count} files")

    def _rename_files_sequentially(self, files: list[Path], progress_callback=None) -> list[Path]:
        """Rename files sequentially regardless of extension, ensuring unique numbers."""
        import uuid  # Add this at the top of the file if not already imported

        self._log("Starting sequential file renaming...")

        # Sort all files to ensure consistent ordering
        sorted_files = sorted(files, key=lambda x: x.name)
        total_files = len(sorted_files)

        # Update initial progress
        if progress_callback:
            progress_callback(0.05)  # Starting progress

        # Check if files are already sequentially named
        needs_renaming = False

        # Track used indices to ensure no duplicates across different extensions
        used_indices = set()
        expected_index = 1

        # First check - each file should have a numeric name starting from 1 with no gaps or duplicates
        for i, file in enumerate(sorted_files):
            # Update progress during check phase (use 10% of total progress)
            if progress_callback and i % 10 == 0:
                progress_callback(0.05 + (i / total_files * 0.1))

            # If filename is not a digit, renaming is needed
            if not file.stem.isdigit():
                self._log(f"Non-numeric filename found: {file.name}. Renaming needed.")
                needs_renaming = True
                break

            # Get current index
            current_index = int(file.stem)

            # Check for duplicates across extensions
            if current_index in used_indices:
                self._log(f"Duplicate number found: {current_index}. Renaming needed.")
                needs_renaming = True
                break

            # Check for proper sequence
            if current_index != expected_index:
                self._log(f"Gap in sequence found: expected {expected_index} but found {current_index}. Renaming needed.")
                needs_renaming = True
                break

            # Mark index as used
            used_indices.add(current_index)
            expected_index += 1

        # If everything is already well-ordered, don't rename
        if not needs_renaming:
            self._log("Files are already properly sequentially named from 1 without gaps or duplicates. No renaming needed.")
            if progress_callback:
                progress_callback(1.0)  # Complete
            return sorted_files

        # If we need to rename, use a two-pass approach
        renamed_files = []

        # STEP 1: Give all files temporary unique names using UUID to avoid conflicts
        temp_files = []
        for i, file in enumerate(sorted_files):
            # Update progress for first pass (use 40% of total progress)
            if progress_callback:
                progress_callback(0.15 + (i / total_files * 0.4))
            else:
                self._update_status(f"Renaming (pass 1/2): {file.name}", (i + 1) / total_files * 0.5)

            ext = file.suffix.lower()
            # Create a temporary name with UUID (using uuid4 for better uniqueness)
            temp_name = file.parent / f"temp_{uuid.uuid4().hex}{ext}"

            try:
                file.rename(temp_name)
                temp_files.append(temp_name)
            except Exception as e:
                self._log(f"Error in temporary rename {file.name}: {e}")
                temp_files.append(file)  # Keep original if rename fails

        # STEP 2: Rename files sequentially across all extensions
        # Sort temp files to maintain original order as much as possible
        temp_files.sort(key=lambda x: x.name)
        next_index = 1

        for i, temp_file in enumerate(temp_files):
            # Update progress for second pass (use 45% of total progress)
            if progress_callback:
                progress_callback(0.55 + (i / len(temp_files) * 0.45))
            else:
                self._update_status(f"Renaming (pass 2/2): {temp_file.name}", 0.5 + (i + 1) / len(temp_files) * 0.5)

            ext = temp_file.suffix.lower()
            new_name = temp_file.parent / f"{next_index}{ext}"

            try:
                temp_file.rename(new_name)
                self._log(f"Final rename: {temp_file.name} → {new_name.name}")
                renamed_files.append(new_name)
                next_index += 1
            except Exception as e:
                self._log(f"Error in final rename {temp_file.name}: {e}")
                renamed_files.append(temp_file)

        # Final progress update
        if progress_callback:
            progress_callback(1.0)

        self._log(f"Sequential renaming complete: {len(renamed_files)} files renamed")
        return renamed_files

    def _resize_large_images(self, files: list[Path], progress_callback=None) -> None:
        """Resize images larger than 4MP (4 million pixels)."""
        self._log("Starting resizing of large images...")

        MAX_MEGAPIXELS = 4_194_304

        # Find image files
        image_files = [
            f for f in files
            if path_util.is_supported_image_extension(f.suffix)
        ]

        self._log(f"Found {len(image_files)} image files to check")
        resized_count = 0

        # Process each image file
        total_files = len(image_files)
        for idx, file in enumerate(image_files):
            # Update progress using callback if available
            if progress_callback:
                progress_callback((idx + 1) / total_files)
            else:
                progress = (idx + 1) / total_files
                self._update_status(f"Analyzing image: {file.name}", progress)

            # Use imagesize to check dimensions (much faster than opening with PIL)
            try:
                width, height = imagesize.get(file)
                total_pixels = width * height
            except Exception as e:
                self._log(f"Error getting dimensions for {file.name}: {e}")
                continue

            # Check if image is larger than 4MP
            if total_pixels > MAX_MEGAPIXELS:
                # Update status to show we're actually resizing
                if not progress_callback:
                    self._update_status(f"Resizing image: {file.name}", (idx + 1) / total_files)

                # Calculate the scaling factor to get to 4MP
                scale_factor = (MAX_MEGAPIXELS / total_pixels) ** 0.5

                # Calculate new dimensions
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)

                self._log(f"Resizing {file.name} from {width}x{height} ({total_pixels/1_000_000:.2f}MP) to {new_width}x{new_height} ({MAX_MEGAPIXELS/1_000_000:.2f}MP)")

                # Now open with PIL to perform the resize
                try:
                    with Image.open(file) as img:
                        # Perform the resize with high quality
                        resized_img = img.resize((new_width, new_height), Image.LANCZOS)
                        # Save with original format and metadata
                        resized_img.save(file, quality=95)
                    resized_count += 1
                except Exception as e:
                    self._log(f"Error resizing {file.name}: {e}")
            else:
                self._log(f"Skipping {file.name}: already under 4MP ({total_pixels/1_000_000:.2f}MP)")

        self._log(f"Completed resizing: {resized_count} images were resized to 4MP")

    def _optimize_pngs(self, files: list[Path], progress_callback=None) -> None:
        """Optimize PNG files using oxipng."""
        self._log("Starting PNG optimization...")

        # Find PNG files
        png_files = [f for f in files if f.suffix.lower() == ".png"]

        if not png_files:
            self._log("No PNG files found to optimize")
            return

        self._log(f"Found {len(png_files)} PNG files to optimize")

        # Track successful optimizations
        success_count = 0
        total_bytes_saved = 0

        # For each PNG file, optimize it
        total_files = len(png_files)
        for idx, file in enumerate(png_files):
            # Update progress using callback if available
            if progress_callback:
                progress_callback((idx + 1) / total_files)
            else:
                progress = (idx + 1) / total_files
                self._update_status(f"Optimizing PNG: {file.name}", progress)

            # Get original file size - might fail if file doesn't exist
            try:
                original_size = file.stat().st_size
            except Exception as e:
                self._log(f"Error getting original size for {file.name}: {e}")
                continue

            # Optimize the PNG file - most likely to fail
            try:
                self._log(f"Optimizing {file.name} with oxipng level 5, fix_errors=True")
                oxipng.optimize(file, level=5, fix_errors=True)
            except Exception as e:
                self._log(f"Error optimizing {file.name}: {e}")
                continue

            # Get new file size and calculate savings
            try:
                new_size = file.stat().st_size
                bytes_saved = original_size - new_size
                percent_saved = (bytes_saved / original_size) * 100 if original_size > 0 else 0

                # Log result
                self._log(f"Optimized {file.name}: {bytes_saved:,} bytes saved ({percent_saved:.1f}%)")

                success_count += 1
                total_bytes_saved += bytes_saved
            except Exception as e:
                self._log(f"Error calculating optimization results for {file.name}: {e}")

        # Log summary
        if success_count > 0:
            avg_saving = total_bytes_saved / success_count
            self._log(f"Completed optimization: {success_count} of {total_files} PNG files optimized")
            self._log(f"Total bytes saved: {total_bytes_saved:,} (average: {avg_saving:,.1f} per file)")
        else:
            self._log("PNG optimization completed with 0 successful optimizations")

    def _convert_to_webp(self, files: list[Path], progress_callback=None) -> None:
        """Convert images to WebP format with optimized settings.

        - PNG, TIFF, and BMP files are converted using lossless WebP format
        - Other formats use lossy compression at 90% quality
        - Original files are only replaced if the WebP version is smaller
        """
        self._log("Starting conversion to WebP format...")

        # Find suitable image files (exclude WebP, JPEG XL, AVIF)
        skip_extensions = ['.webp', '.jxl', '.avif']
        image_files = [
            f for f in files
            if path_util.is_supported_image_extension(f.suffix) and
            f.suffix.lower() not in skip_extensions
        ]

        if not image_files:
            self._log("No suitable image files found to convert")
            return

        self._log(f"Found {len(image_files)} images to convert to WebP")

        # Track conversion statistics
        success_count = 0
        total_bytes_saved = 0
        skipped_count = 0

        # Define which formats should use lossless encoding
        lossless_extensions = ['.png', '.tiff', '.tif', '.bmp']

        # Process each image file
        total_files = len(image_files)
        for idx, file in enumerate(image_files):
            # Update progress (outside try-except)
            if progress_callback:
                progress_callback((idx + 1) / total_files)
            else:
                progress = (idx + 1) / total_files
                self._update_status(f"Converting to WebP: {file.name}", progress)

            # Skip files that don't exist or can't be accessed
            try:
                original_size = file.stat().st_size
                webp_path = file.with_suffix('.webp')
            except Exception as e:
                self._log(f"Error accessing {file.name}: {e}")
                continue

            # Process the image conversion
            try:
                with Image.open(file) as img:
                    # Determine if this is a format that should use lossless encoding
                    is_lossless_candidate = file.suffix.lower() in lossless_extensions

                    # Convert to RGB if the image is in RGBA mode and has no transparency
                    if img.mode == 'RGBA':
                        # Check if the image has any transparency
                        if not any(px[3] < 255 for px in img.getdata()):
                            img = img.convert('RGB')
                            is_lossless_candidate = False  # Treat as RGB image now

                    # Log conversion info with encoding method
                    if is_lossless_candidate:
                        self._log(f"Converting {file.name} to WebP using lossless encoding")
                        img.save(webp_path, 'WEBP', lossless=True)
                    else:
                        self._log(f"Converting {file.name} to WebP using 90% quality")
                        img.save(webp_path, 'WEBP', quality=90)
            except Exception as e:
                self._log(f"Error converting {file.name} to WebP: {e}")
                # Try to clean up partial WebP file if it exists
                if webp_path.exists():
                    # Use contextlib.suppress instead of try-except-pass
                    with contextlib.suppress(FileNotFoundError, PermissionError):
                        webp_path.unlink()
                continue

            # Handle file operations separately
            try:
                if webp_path.exists():
                    # Get new file size and calculate savings
                    new_size = webp_path.stat().st_size
                    bytes_saved = original_size - new_size
                    percent_saved = (bytes_saved / original_size) * 100 if original_size > 0 else 0

                    # Only delete original if the WebP file is smaller
                    if new_size < original_size:
                        file.unlink()
                        self._log(f"Converted {file.name} to WebP: {bytes_saved:,} bytes saved ({percent_saved:.1f}%)")
                        success_count += 1
                        total_bytes_saved += bytes_saved
                    else:
                        # WebP is larger, so keep original and delete WebP
                        webp_path.unlink()
                        self._log(f"Skipped {file.name}: WebP version would be larger by {-bytes_saved:,} bytes")
                        skipped_count += 1
            except Exception as e:
                self._log(f"Error finalizing conversion for {file.name}: {e}")

        # Log summary
        if success_count > 0:
            avg_saving = total_bytes_saved / success_count
            self._log(f"Completed WebP conversion: {success_count} of {total_files} images converted")
            self._log(f"Total bytes saved: {total_bytes_saved:,} (average: {avg_saving:,.1f} per file)")
            if skipped_count > 0:
                self._log(f"Skipped {skipped_count} images where WebP would be larger")
        else:
            self._log("WebP conversion completed with 0 successful conversions")

    def _convert_to_jpegxl(self, files: list[Path], progress_callback=None) -> None:
        """Convert images to JPEG XL format.

        - JPEG/JPG files are converted losslessly (perfect quality preservation)
        - Other formats use lossy compression at 90% quality
        - Original files are only replaced if the JXL version is smaller
        """
        self._log("Starting conversion to JPEG XL format...")

        # Find suitable image files (excluding JXL files)
        image_files = [
            f for f in files
            if path_util.is_supported_image_extension(f.suffix) and
            f.suffix.lower() != '.jxl'
        ]

        if not image_files:
            self._log("No image files found to convert to JPEG XL")
            return

        self._log(f"Found {len(image_files)} images to convert to JPEG XL")

        # Track conversion statistics
        success_count = 0
        total_bytes_saved = 0
        skipped_count = 0

        # Process each image file
        total_files = len(image_files)
        for idx, file in enumerate(image_files):
            # Update progress (outside try-except)
            if progress_callback:
                progress_callback((idx + 1) / total_files)
            else:
                progress = (idx + 1) / total_files
                self._update_status(f"Converting to JPEG XL: {file.name}", progress)

            # Skip files that don't exist or can't be accessed
            try:
                original_size = file.stat().st_size
                jxl_path = file.with_suffix('.jxl')
            except Exception as e:
                self._log(f"Error accessing {file.name}: {e}")
                continue

            # Process the image conversion
            try:
                # Determine if this is a JPEG file (for lossless transcoding)
                is_jpeg = file.suffix.lower() in ['.jpg', '.jpeg']

                # Open the image with PIL
                with Image.open(file) as img:
                    if is_jpeg:
                        # Use lossless transcoding for JPEG files
                        self._log(f"Converting {file.name} to JPEG XL using lossless encoding")
                        img.save(jxl_path, 'JXL', lossless=True)
                    else:
                        # Use lossy encoding at 90% quality for other formats
                        self._log(f"Converting {file.name} to JPEG XL using lossy compression (90% quality)")
                        img.save(jxl_path, 'JXL', quality=90)
            except Exception as e:
                self._log(f"Error converting {file.name} to JPEG XL: {e}")
                # Try to clean up partial JXL file if it exists
                if jxl_path.exists():
                    with contextlib.suppress(FileNotFoundError, PermissionError):
                        jxl_path.unlink()
                continue

            # Handle file operations separately
            try:
                if jxl_path.exists():
                    # Get new file size and calculate savings
                    new_size = jxl_path.stat().st_size
                    bytes_saved = original_size - new_size
                    percent_saved = (bytes_saved / original_size) * 100 if original_size > 0 else 0

                    # Only delete original if the JXL file is smaller
                    if new_size < original_size:
                        file.unlink()
                        self._log(f"Converted {file.name} to JPEG XL: {bytes_saved:,} bytes saved ({percent_saved:.1f}%)")
                        success_count += 1
                        total_bytes_saved += bytes_saved
                    else:
                        # JXL is larger, so keep original and delete JXL
                        jxl_path.unlink()
                        self._log(f"Skipped {file.name}: JPEG XL version would be larger by {-bytes_saved:,} bytes")
                        skipped_count += 1
            except Exception as e:
                self._log(f"Error finalizing conversion for {file.name}: {e}")

        # Log summary
        if success_count > 0:
            avg_saving = total_bytes_saved / success_count
            self._log(f"Completed JPEG XL conversion: {success_count} of {total_files} images converted")
            self._log(f"Total bytes saved: {total_bytes_saved:,} (average: {avg_saving:,.1f} per file)")
            if skipped_count > 0:
                self._log(f"Skipped {skipped_count} images where JPEG XL would be larger")
        else:
            self._log("JPEG XL conversion completed with 0 successful conversions")
