from tkinter import filedialog

import customtkinter as ctk


class GenerateMasksWindow(ctk.CTkToplevel):
    def __init__(self, parent, path, parent_include_subdirectories, *args, **kwargs):
        """
        Window for generating masks for a folder of images

        Parameters:
            parent (`Tk`): the parent window
            path (`str`): the path to the folder
            parent_include_subdirectories (`bool`): whether to include subdirectories. used to set the default value of the include subdirectories checkbox
        """
        ctk.CTkToplevel.__init__(self, parent, *args, **kwargs)
        self.parent = parent

        if path is None:
            path = ""

        self.title("Batch generate masks")
        self.geometry("360x430")
        self.resizable(False, False)
        self.wait_visibility()
        self.grab_set()
        self.focus_set()

        self.mode_var = ctk.StringVar(self, "Create if absent")
        self.modes = ["Replace all masks", "Create if absent", "Add to existing", "Subtract from existing", "Blend with existing"]
        self.model_var = ctk.StringVar(self, "ClipSeg")
        self.models = ["ClipSeg", "Rembg", "Rembg-Human", "Hex Color"]

        self.frame = ctk.CTkFrame(self, width=600, height=300)
        self.frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        self.model_label = ctk.CTkLabel(self.frame, text="Model", width=100)
        self.model_label.grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.model_dropdown = ctk.CTkOptionMenu(self.frame, variable=self.model_var, values=self.models, dynamic_resizing=False, width=200)
        self.model_dropdown.grid(row=0, column=1, sticky="w", padx=5, pady=5)

        self.path_label = ctk.CTkLabel(self.frame, text="Folder", width=100)
        self.path_label.grid(row=1, column=0, sticky="w",padx=5, pady=5)
        self.path_entry = ctk.CTkEntry(self.frame, width=150)
        self.path_entry.insert(0, path)
        self.path_entry.grid(row=1, column=1, sticky="w", padx=5, pady=5)
        self.path_button = ctk.CTkButton(self.frame, width=30, text="...", command=lambda: self.browse_for_path(self.path_entry))
        self.path_button.grid(row=1, column=1, sticky="e", padx=5, pady=5)

        self.prompt_label = ctk.CTkLabel(self.frame, text="Prompt", width=100)
        self.prompt_label.grid(row=2, column=0, sticky="w",padx=5, pady=5)
        self.prompt_entry = ctk.CTkEntry(self.frame, width=200)
        self.prompt_entry.grid(row=2, column=1, sticky="w", padx=5, pady=5)

        self.mode_label = ctk.CTkLabel(self.frame, text="Mode", width=100)
        self.mode_label.grid(row=3, column=0, sticky="w", padx=5, pady=5)
        self.mode_dropdown = ctk.CTkOptionMenu(self.frame, variable=self.mode_var, values=self.modes, dynamic_resizing=False, width=200)
        self.mode_dropdown.grid(row=3, column=1, sticky="w", padx=5, pady=5)

        self.threshold_label = ctk.CTkLabel(self.frame, text="Threshold", width=100)
        self.threshold_label.grid(row=4, column=0, sticky="w", padx=5, pady=5)
        self.threshold_entry = ctk.CTkEntry(self.frame, width=200, placeholder_text="0.0 - 1.0")
        self.threshold_entry.insert(0, "0.3")
        self.threshold_entry.grid(row=4, column=1, sticky="w", padx=5, pady=5)

        self.smooth_label = ctk.CTkLabel(self.frame, text="Smooth", width=100)
        self.smooth_label.grid(row=5, column=0, sticky="w", padx=5, pady=5)
        self.smooth_entry = ctk.CTkEntry(self.frame, width=200, placeholder_text="5")
        self.smooth_entry.insert(0, 5)
        self.smooth_entry.grid(row=5, column=1, sticky="w", padx=5, pady=5)

        self.expand_label = ctk.CTkLabel(self.frame, text="Expand", width=100)
        self.expand_label.grid(row=6, column=0, sticky="w", padx=5, pady=5)
        self.expand_entry = ctk.CTkEntry(self.frame, width=200, placeholder_text="10")
        self.expand_entry.insert(0, 10)
        self.expand_entry.grid(row=6, column=1, sticky="w", padx=5, pady=5)

        self.alpha_label = ctk.CTkLabel(self.frame, text="Alpha", width=100)
        self.alpha_label.grid(row=7, column=0, sticky="w", padx=5, pady=5)
        self.alpha_entry = ctk.CTkEntry(self.frame, width=200, placeholder_text="1")
        self.alpha_entry.insert(0, 1)
        self.alpha_entry.grid(row=7, column=1, sticky="w", padx=5, pady=5)

        self.include_subdirectories_label = ctk.CTkLabel(self.frame, text="Include subfolders", width=100)
        self.include_subdirectories_label.grid(row=8, column=0, sticky="w", padx=5, pady=5)
        self.include_subdirectories_var = ctk.BooleanVar(self, parent_include_subdirectories)
        self.include_subdirectories_switch = ctk.CTkSwitch(self.frame, text="", variable=self.include_subdirectories_var)
        self.include_subdirectories_switch.grid(row=8, column=1, sticky="w", padx=5, pady=5)

        self.progress_label = ctk.CTkLabel(self.frame, text="Progress: 0/0", width=100)
        self.progress_label.grid(row=9, column=0, sticky="w", padx=5, pady=5)
        self.progress = ctk.CTkProgressBar(self.frame, orientation="horizontal", mode="determinate", width=200)
        self.progress.grid(row=9, column=1, sticky="w", padx=5, pady=5)

        self.create_masks_button = ctk.CTkButton(self.frame, text="Create Masks", width=310, command=self.create_masks)
        self.create_masks_button.grid(row=10, column=0, columnspan=2, sticky="w", padx=5, pady=5)

        self.frame.pack(fill="both", expand=True)

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
        self.progress_label.configure(text="{0}/{1}".format(value, max_value))
        self.progress.update()

    def create_masks(self):
        self.parent.load_masking_model(self.model_var.get())

        mode = {
            "Replace all masks": "replace",
            "Create if absent": "fill",
            "Add to existing": "add",
            "Subtract from existing": "subtract",
            "Blend with existing": "blend",
        }[self.mode_var.get()]

        self.parent.masking_model.mask_folder(
            sample_dir=self.path_entry.get(),
            prompts=[self.prompt_entry.get()],
            mode=mode,
            alpha=float(self.alpha_entry.get()),
            threshold=float(self.threshold_entry.get()),
            smooth_pixels=int(self.smooth_entry.get()),
            expand_pixels=int(self.expand_entry.get()),
            progress_callback=self.set_progress,
            include_subdirectories=self.include_subdirectories_var.get(),
        )
        self.parent.load_image()
