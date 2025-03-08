from tkinter import filedialog

import customtkinter as ctk


class GenerateCaptionsWindow(ctk.CTkToplevel):
    def __init__(
        self, parent, path, parent_include_subdirectories, *args, **kwargs
    ):
        ctk.CTkToplevel.__init__(self, parent, *args, **kwargs)
        self.parent = parent

        # Setup window properties
        self._setup_window("Batch generate captions", "360x360")

        # Get available models dynamically
        self.models = (
            self.parent.model_manager.get_available_captioning_models()
        )
        self.model_var = ctk.StringVar(
            self, self.models[0] if self.models else ""
        )

        # Define modes with mapping to API values
        self.modes = [
            "Replace all captions",
            "Create if absent",
            "Add as new line",
        ]
        self.mode_mapping = {
            "Replace all captions": "replace",
            "Create if absent": "fill",
            "Add as new line": "add",
        }
        self.mode_var = ctk.StringVar(self, "Create if absent")

        # Set up the UI
        self._create_layout(path, parent_include_subdirectories)

    def _setup_window(self, title, geometry):
        self.title(title)
        self.geometry(geometry)
        self.resizable(True, True)
        self.wait_visibility()
        self.grab_set()
        self.focus_set()

    def _create_layout(self, path, parent_include_subdirectories):
        # Create frame
        self.frame = ctk.CTkFrame(self, width=600, height=300)
        self.frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        # Create UI components
        self._create_model_selection()
        self._create_path_selection(path)
        self._create_caption_configuration()
        self._create_mode_selection()
        self._create_subdirectory_option(parent_include_subdirectories)
        self._create_progress_indicators()
        self._create_action_buttons()

        self.frame.pack(fill="both", expand=True)

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

    def _create_caption_configuration(self):
        # Initial caption
        self.caption_label = ctk.CTkLabel(self.frame, text="Initial Caption", width=100)
        self.caption_label.grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.caption_entry = ctk.CTkEntry(self.frame, width=200)
        self.caption_entry.grid(row=2, column=1, sticky="w", padx=5, pady=5)

        # Prefix
        self.prefix_label = ctk.CTkLabel(self.frame, text="Caption Prefix", width=100)
        self.prefix_label.grid(row=3, column=0, sticky="w", padx=5, pady=5)
        self.prefix_entry = ctk.CTkEntry(self.frame, width=200)
        self.prefix_entry.grid(row=3, column=1, sticky="w", padx=5, pady=5)

        # Postfix
        self.postfix_label = ctk.CTkLabel(self.frame, text="Caption Postfix", width=100)
        self.postfix_label.grid(row=4, column=0, sticky="w", padx=5, pady=5)
        self.postfix_entry = ctk.CTkEntry(self.frame, width=200)
        self.postfix_entry.grid(row=4, column=1, sticky="w", padx=5, pady=5)

    def _create_mode_selection(self):
        self.mode_label = ctk.CTkLabel(self.frame, text="Mode", width=100)
        self.mode_label.grid(row=5, column=0, sticky="w", padx=5, pady=5)
        self.mode_dropdown = ctk.CTkOptionMenu(self.frame, variable=self.mode_var, values=self.modes, dynamic_resizing=False, width=200)
        self.mode_dropdown.grid(row=5, column=1, sticky="w", padx=5, pady=5)

    def _create_subdirectory_option(self, parent_include_subdirectories):
        self.include_subdirectories_label = ctk.CTkLabel(self.frame, text="Include subfolders", width=100)
        self.include_subdirectories_label.grid(row=6, column=0, sticky="w", padx=5, pady=5)
        self.include_subdirectories_var = ctk.BooleanVar(self, parent_include_subdirectories)
        self.include_subdirectories_switch = ctk.CTkSwitch(self.frame, text="", variable=self.include_subdirectories_var)
        self.include_subdirectories_switch.grid(row=6, column=1, sticky="w", padx=5, pady=5)

    def _create_progress_indicators(self):
        self.progress_label = ctk.CTkLabel(self.frame, text="Progress: 0/0", width=100)
        self.progress_label.grid(row=7, column=0, sticky="w", padx=5, pady=5)
        self.progress = ctk.CTkProgressBar(self.frame, orientation="horizontal", mode="determinate", width=200)
        self.progress.grid(row=7, column=1, sticky="w", padx=5, pady=5)

    def _create_action_buttons(self):
        self.create_captions_button = ctk.CTkButton(self.frame, text="Create Captions", width=310, command=self.create_captions)
        self.create_captions_button.grid(row=8, column=0, columnspan=2, sticky="w", padx=5, pady=5)

    def browse_for_path(self, entry_box):
        # get the path from the user
        path = filedialog.askdirectory()
        # set the path to the entry box
        # delete entry box text
        entry_box.focus_set()
        entry_box.delete(0, filedialog.END)
        entry_box.insert(0, path)
        self.focus_set()

    def set_progress(self, value, max_value):
        progress = value / max_value
        self.progress.set(progress)
        self.progress_label.configure(text=f"{value}/{max_value}")
        self.progress.update()

    def create_captions(self):
        captioning_model = self.parent.model_manager.load_captioning_model(
            self.model_var.get()
        )

        # Skip processing if model failed to load
        if captioning_model is None:
            print(
                f"Failed to load captioning model: {self.model_var.get()}"
            )
            return

        mode = self.mode_mapping[self.mode_var.get()]

        captioning_model.caption_folder(
            sample_dir=self.path_entry.get(),
            initial_caption=self.caption_entry.get(),
            caption_prefix=self.prefix_entry.get(),
            caption_postfix=self.postfix_entry.get(),
            mode=mode,
            progress_callback=self.set_progress,
            include_subdirectories=self.include_subdirectories_var.get(),
        )

        # Reload the current image data using image_handler
        if hasattr(self.parent, "image_handler") and hasattr(
            self.parent.image_handler, "load_image_data"
        ):
            self.parent.image_handler.load_image_data()
            self.parent.refresh_ui()
