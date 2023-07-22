from tkinter import filedialog

import customtkinter as ctk


class GenerateMasksWindow(ctk.CTkToplevel):
    def __init__(self, parent, path, *args, **kwargs):
        ctk.CTkToplevel.__init__(self, parent, *args, **kwargs)
        self.parent = parent

        self.title("Batch generate masks")
        self.geometry("320x310")
        self.resizable(False, False)
        self.wait_visibility()
        self.grab_set()
        self.focus_set()

        self.mode_var = ctk.StringVar(self, "Create if absent")
        self.modes = ["Replace all masks", "Create if absent", "Add to existing", "Subtract from existing"]

        self.frame = ctk.CTkFrame(self, width=600, height=300)
        self.frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        self.path_label = ctk.CTkLabel(self.frame, text="Folder", width=100)
        self.path_label.grid(row=0, column=0, sticky="w",padx=5, pady=5)
        self.path_entry = ctk.CTkEntry(self.frame, width=150)
        self.path_entry.insert(0, path)
        self.path_entry.grid(row=0, column=1, sticky="w", padx=5, pady=5)
        self.path_button = ctk.CTkButton(self.frame, width=30, text="...", command=lambda: self.browse_for_path(self.path_entry))
        self.path_button.grid(row=0, column=1, sticky="e", padx=5, pady=5)

        self.prompt_label = ctk.CTkLabel(self.frame, text="Prompt", width=100)
        self.prompt_label.grid(row=1, column=0, sticky="w",padx=5, pady=5)
        self.prompt_entry = ctk.CTkEntry(self.frame, width=200)
        self.prompt_entry.grid(row=1, column=1, sticky="w", padx=5, pady=5)

        self.mode_label = ctk.CTkLabel(self.frame, text="Mode", width=100)
        self.mode_label.grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.mode_dropdown = ctk.CTkOptionMenu(self.frame, variable=self.mode_var, values=self.modes, dynamic_resizing=False, width=200)
        self.mode_dropdown.grid(row=2, column=1, sticky="w", padx=5, pady=5)

        self.threshold_label = ctk.CTkLabel(self.frame, text="Threshold", width=100)
        self.threshold_label.grid(row=3, column=0, sticky="w", padx=5, pady=5)
        self.threshold_entry = ctk.CTkEntry(self.frame, width=200, placeholder_text="0.0 - 1.0")
        self.threshold_entry.insert(0, "0.3")
        self.threshold_entry.grid(row=3, column=1, sticky="w", padx=5, pady=5)

        self.smooth_label = ctk.CTkLabel(self.frame, text="Smooth", width=100)
        self.smooth_label.grid(row=4, column=0, sticky="w", padx=5, pady=5)
        self.smooth_entry = ctk.CTkEntry(self.frame, width=200, placeholder_text="5")
        self.smooth_entry.insert(0, 5)
        self.smooth_entry.grid(row=4, column=1, sticky="w", padx=5, pady=5)

        self.expand_label = ctk.CTkLabel(self.frame, text="Expand", width=100)
        self.expand_label.grid(row=5, column=0, sticky="w", padx=5, pady=5)
        self.expand_entry = ctk.CTkEntry(self.frame, width=200, placeholder_text="10")
        self.expand_entry.insert(0, 10)
        self.expand_entry.grid(row=5, column=1, sticky="w", padx=5, pady=5)

        self.progress_label = ctk.CTkLabel(self.frame, text="Progress: 0/0", width=100)
        self.progress_label.grid(row=6, column=0, sticky="w", padx=5, pady=5)
        self.progress = ctk.CTkProgressBar(self.frame, orientation="horizontal", mode="determinate", width=200)
        self.progress.grid(row=6, column=1, sticky="w", padx=5, pady=5)

        self.create_masks_button = ctk.CTkButton(self.frame, text="Create Masks", width=310, command=self.create_masks)
        self.create_masks_button.grid(row=7, column=0, columnspan=2, sticky="w", padx=5, pady=5)

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
        self.parent.load_masking_model()

        mode = {
            "Replace all masks": "replace",
            "Create if absent": "fill",
            "Add to existing": "add",
            "Subtract from existing": "subtract"
        }[self.mode_var.get()]

        self.parent.masking_model.mask_folder(
            sample_dir=self.path_entry.get(),
            prompts=[self.prompt_entry.get()],
            mode=mode,
            threshold=float(self.threshold_entry.get()),
            smooth_pixels=int(self.smooth_entry.get()),
            expand_pixels=int(self.expand_entry.get()),
            progress_callback=self.set_progress,
        )
        self.parent.load_image()
