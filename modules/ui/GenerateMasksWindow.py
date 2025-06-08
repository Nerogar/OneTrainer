import logging
import threading
from tkinter import END, filedialog, messagebox

from modules.util.ui.ToolTip import ToolTip
from modules.util.ui.ui_utils import set_window_icon

import customtkinter as ctk

logger = logging.getLogger(__name__)

class GenerateMasksWindow(ctk.CTkToplevel):
    def __init__(
            self, parent, path, include_subdirectories, *args, **kwargs
        ):
            """
            Window for generating masks for a folder of images

            Parameters:
                parent (`Tk`): the parent window
                path (`str`): the path to the folder
                include_subdirectories (`bool`): whether to include subdirectories
            """
            super().__init__(parent, *args, **kwargs)
            self.parent = parent

            if path is None:
                path = ""

            # Setup window properties
            self._setup_window("Batch generate masks", "360x480")

            # Get available models dynamically
            self.models = self.parent.model_manager.get_available_masking_models()
            self.model_var = ctk.StringVar(
                self, self.models[0] if self.models else ""
            )

            # Define modes with mapping to API values
            self.modes = [
                "Replace all masks",
                "Create if absent",
                "Add to existing",
                "Subtract from existing",
                "Blend with existing",
            ]
            self.mode_var = ctk.StringVar(self, "Create if absent")

            # Add preview mode toggle
            self.preview_mode_var = ctk.BooleanVar(self, False)

            # Set up the UI
            self._create_layout(path, include_subdirectories)

    def _setup_window(self, title, geometry):
        self.title(title)
        self.geometry(geometry)
        self.resizable(True, True)
        self.wait_visibility()
        self.grab_set()
        self.focus_set()
        self.after(200, lambda: set_window_icon(self))

    def _create_layout(self, path, include_subdirectories):
        # Create frame
        self.frame = ctk.CTkFrame(self, width=600, height=340)
        self.frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        # Configure the root window grid to allow frame expansion
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Create UI components
        self._create_model_selection()
        self._create_path_selection(path)
        self._create_mask_options()
        self._create_subdirectory_option(include_subdirectories)
        self._create_preview_option()
        self._create_progress_indicators()
        self._create_action_buttons()

    def _create_model_selection(self):
        self.model_label = ctk.CTkLabel(self.frame, text="Model", width=100)
        self.model_label.grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.model_dropdown = ctk.CTkOptionMenu(self.frame, variable=self.model_var, values=self.models, dynamic_resizing=False, width=200)
        self.model_dropdown.grid(row=0, column=1, sticky="w", padx=5, pady=5)

    def _create_path_selection(self, path):
        self.path_label = ctk.CTkLabel(self.frame, text="Folder", width=100)
        self.path_label.grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.path_entry = ctk.CTkEntry(self.frame, width=150)
        self.path_entry.insert(0, path)
        self.path_entry.grid(row=1, column=1, sticky="w", padx=5, pady=5)
        self.path_button = ctk.CTkButton(self.frame, width=30, text="...", command=lambda: self.browse_for_path(self.path_entry))
        self.path_button.grid(row=1, column=1, sticky="e", padx=5, pady=5)

    def _create_mask_options(self):
        # Prompt
        self.prompt_label = ctk.CTkLabel(self.frame, text="Prompt", width=100)
        self.prompt_label.grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.prompt_entry = ctk.CTkEntry(self.frame, width=200)
        self.prompt_entry.grid(row=2, column=1, sticky="w", padx=5, pady=5)
        ToolTip(self.prompt_entry, "Enter object to detect (e.g. 'person', 'dog', 'car')")

        # Mode (moved up for better flow)
        self.mode_label = ctk.CTkLabel(self.frame, text="Mode", width=100)
        self.mode_label.grid(row=3, column=0, sticky="w", padx=5, pady=5)
        self.mode_dropdown = ctk.CTkOptionMenu(self.frame, variable=self.mode_var, values=self.modes, dynamic_resizing=False, width=200)
        self.mode_dropdown.grid(row=3, column=1, sticky="w", padx=5, pady=5)

        # Threshold
        self.threshold_label = ctk.CTkLabel(self.frame, text="Threshold", width=100)
        self.threshold_label.grid(row=4, column=0, sticky="w", padx=5, pady=5)
        self.threshold_entry = ctk.CTkEntry(self.frame, width=200, placeholder_text="0.0 - 1.0")
        self.threshold_entry.insert(0, "0.3")
        self.threshold_entry.grid(row=4, column=1, sticky="w", padx=5, pady=5)
        ToolTip(self.threshold_entry, "Confidence threshold: Lower values detect more objects but may include incorrect regions")

        # Smooth - Changed default to 0 since SAM2 has built-in smoothing
        self.smooth_label = ctk.CTkLabel(self.frame, text="Smooth", width=100)
        self.smooth_label.grid(row=5, column=0, sticky="w", padx=5, pady=5)
        self.smooth_entry = ctk.CTkEntry(self.frame, width=200, placeholder_text="0-10")
        self.smooth_entry.insert(0, "0")  # Changed default to 0
        self.smooth_entry.grid(row=5, column=1, sticky="w", padx=5, pady=5)
        ToolTip(self.smooth_entry, "Additional smoothing (0=use built-in smoothing, higher values for extra smoothing)")

        # Expand
        self.expand_label = ctk.CTkLabel(self.frame, text="Expand", width=100)
        self.expand_label.grid(row=6, column=0, sticky="w", padx=5, pady=5)
        self.expand_entry = ctk.CTkEntry(self.frame, width=200, placeholder_text="0-20")
        self.expand_entry.insert(0, "10")
        self.expand_entry.grid(row=6, column=1, sticky="w", padx=5, pady=5)
        ToolTip(self.expand_entry, "Expansion pixels: Expands mask boundaries outward")

        # Alpha
        self.alpha_label = ctk.CTkLabel(self.frame, text="Alpha", width=100)
        self.alpha_label.grid(row=7, column=0, sticky="w", padx=5, pady=5)
        self.alpha_entry = ctk.CTkEntry(self.frame, width=200, placeholder_text="0.0 - 1.0")
        self.alpha_entry.insert(0, "1")
        self.alpha_entry.grid(row=7, column=1, sticky="w", padx=5, pady=5)
        ToolTip(self.alpha_entry, "Blending strength when combining with existing masks")

    def _create_subdirectory_option(self, include_subdirectories):
        self.include_subdirectories_label = ctk.CTkLabel(self.frame, text="Include subdirs", width=100)
        self.include_subdirectories_label.grid(row=8, column=0, sticky="w", padx=5, pady=5)
        self.include_subdirectories_var = ctk.BooleanVar(self, include_subdirectories)
        self.include_subdirectories_switch = ctk.CTkSwitch(self.frame, text="", variable=self.include_subdirectories_var)
        self.include_subdirectories_switch.grid(row=8, column=1, sticky="w", padx=5, pady=5)

    def _create_preview_option(self):
        """Create a toggle for test mode (only process current image)"""
        self.preview_mode_label = ctk.CTkLabel(self.frame, text="Test Run", width=100)
        self.preview_mode_label.grid(row=9, column=0, sticky="w", padx=5, pady=5)
        self.preview_mode_switch = ctk.CTkSwitch(self.frame, text="", variable=self.preview_mode_var)
        self.preview_mode_switch.grid(row=9, column=1, sticky="w", padx=5, pady=5)
        ToolTip(self.preview_mode_switch, "Masks only the current image to more quickly get feedback")

    def _create_progress_indicators(self):
        self.progress_label = ctk.CTkLabel(self.frame, text="Progress: 0/0", width=100)
        self.progress_label.grid(row=10, column=0, sticky="w", padx=5, pady=5)
        self.progress = ctk.CTkProgressBar(self.frame, orientation="horizontal", mode="determinate", width=200)
        self.progress.grid(row=10, column=1, sticky="w", padx=5, pady=5)

    def _create_action_buttons(self):
        self.create_masks_button = ctk.CTkButton(self.frame, text="Create Masks", width=310, command=self.create_masks)
        self.create_masks_button.grid(row=11, column=0, columnspan=2, sticky="w", padx=5, pady=5)

    def browse_for_path(self, entry_box):
        # get the path from the user
        path = filedialog.askdirectory()
        # set the path to the entry box
        # delete entry box text
        entry_box.focus_set()
        entry_box.delete(0, END)  # Using proper END constant
        entry_box.insert(0, path)
        self.focus_set()

    def set_progress(self, value, max_value):
        progress = value / max_value
        self.progress.set(progress)
        self.progress_label.configure(text=f"{value}/{max_value}")
        self.progress.update()

    def create_masks(self):
        try:
            threshold = float(self.threshold_entry.get())
            if not 0 <= threshold <= 0.90:
                messagebox.showwarning("Invalid Value", "Threshold must be between 0.0 and 0.9 for usable results")
                return

            smooth_pixels = int(self.smooth_entry.get())
            if smooth_pixels < 0 or smooth_pixels > 10:
                messagebox.showwarning("Invalid Value", "Smooth pixels should be between 0 and 10")
                return

            expand_pixels = int(self.expand_entry.get())
            if expand_pixels < 0 or expand_pixels > 64:
                messagebox.showwarning("Invalid Value", "Expand pixels should be between 0 and 64")
                return

            alpha = float(self.alpha_entry.get())
            if not 0 <= alpha <= 1:
                messagebox.showwarning("Invalid Value", "Alpha must be between 0.0 and 1.0")
                return

            # Check for empty prompt
            prompt = self.prompt_entry.get().strip()
            if not prompt:
                messagebox.showwarning("Invalid Input", "Please enter a detection prompt")
                return
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numeric values for threshold, smooth, expand, and alpha")
            return

        # Disable the create button to prevent multiple clicks
        self.create_masks_button.configure(state="disabled", text="Processing...")

        # Run the masking process in a background thread
        thread = threading.Thread(target=self._run_masking_process, args=(prompt, alpha, threshold, smooth_pixels, expand_pixels))
        thread.daemon = True  # Thread will close when main application closes
        thread.start()

    def _run_masking_process(self, prompt, alpha, threshold, smooth_pixels, expand_pixels):
        try:
            masking_model = self.parent.model_manager.load_masking_model(
                self.model_var.get()
            )

            # Skip processing if model failed to load
            if masking_model is None:
                print(f"Failed to load masking model: {self.model_var.get()}")
                return

            # Set model_manager reference if this is a SAMdreamMaskModel
            if hasattr(masking_model, "__class__") and hasattr(masking_model.__class__, "__name__"):
                if masking_model.__class__.__name__ == "SAMdreamMaskModel":
                    masking_model.model_manager = self.parent.model_manager

            mode = {
                "Replace all masks": "replace",
                "Create if absent": "fill",
                "Add to existing": "add",
                "Subtract from existing": "subtract",
                "Blend with existing": "blend",
            }[self.mode_var.get()]

            # Check if preview mode is enabled
            if self.preview_mode_var.get() and hasattr(self.parent, "current_image_index"):
                # Get the current image path from parent window
                if (hasattr(self.parent, "image_rel_paths") and
                    0 <= self.parent.current_image_index < len(self.parent.image_rel_paths)):

                    # Get the full path to the current image
                    current_image_rel_path = self.parent.image_rel_paths[self.parent.current_image_index]
                    logger.info(f"Preview mode: Processing only image {current_image_rel_path}")

                    # Process just this single image
                    # We'll pass the sample_dir as the parent directory, but also pass the full relative path
                    # for precise filtering
                    masking_model.mask_folder(
                        sample_dir=self.parent.dir,
                        prompts=[prompt],
                        mode=mode,
                        alpha=alpha,
                        threshold=threshold,
                        smooth_pixels=smooth_pixels,
                        expand_pixels=expand_pixels,
                        progress_callback=self.set_progress,
                        include_subdirectories=True,  # Ensure we can find the file in subdirectories
                        single_file=current_image_rel_path  # Pass the relative path, not just the filename
                    )
                else:
                    messagebox.showwarning("Preview Error", "No current image is selected")
                    self.after(100, lambda: self.create_masks_button.configure(state="normal", text="Create Masks"))
                    return
            else:
                # Normal folder processing
                masking_model.mask_folder(
                    sample_dir=self.path_entry.get(),
                    prompts=[prompt],
                    mode=mode,
                    alpha=alpha,
                    threshold=threshold,
                    smooth_pixels=smooth_pixels,
                    expand_pixels=expand_pixels,
                    progress_callback=self.set_progress,
                    include_subdirectories=self.include_subdirectories_var.get(),
                )

            # Use after to safely update UI from the main thread
            self.after(100, self._update_ui_after_processing)
        except Exception as e:
            error_message = str(e)
            print(f"Error during masking process: {error_message}")
            # Use after to safely update UI from the main thread
            self.after(100, lambda: self._handle_masking_error(error_message))

    def _update_ui_after_processing(self):
        # Re-enable the create button
        self.create_masks_button.configure(state="normal", text="Create Masks")

        # Reload the current image data using image_handler
        if hasattr(self.parent, "image_handler") and hasattr(
            self.parent.image_handler, "load_image_data"
        ):
            self.parent.image_handler.load_image_data()
            self.parent.refresh_ui()

    def _handle_masking_error(self, error_message):
        self.create_masks_button.configure(state="normal", text="Create Masks")
        messagebox.showerror("Processing Error", f"An error occurred during mask creation:\n{error_message}")
