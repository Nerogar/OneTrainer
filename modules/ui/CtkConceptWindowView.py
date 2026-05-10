import threading

from modules.ui.BaseConceptWindowView import BaseConceptWindowView
from modules.ui.ConceptWindowController import ConceptWindowController
from modules.util.ui import ctk_components
from modules.util.ui.ui_utils import set_window_icon

import customtkinter as ctk
from customtkinter import AppearanceModeTracker, ThemeManager
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class CtkConceptWindowView(BaseConceptWindowView, ctk.CTkToplevel):
    def __init__(
            self,
            parent,
            controller: ConceptWindowController,
            ui_state,
            image_ui_state,
            text_ui_state,
            *args, **kwargs,
    ):
        ctk.CTkToplevel.__init__(self, parent, *args, **kwargs)
        BaseConceptWindowView.__init__(self, ctk_components)

        self.controller = controller
        self.image_preview_file_index = 0
        self.preview_augmentations = ctk.BooleanVar(self, True)
        self.bucket_fig = None

        self.title("Concept")
        self.geometry("800x700")
        self.resizable(True, True)

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        tabview = ctk.CTkTabview(self)
        tabview.grid(row=0, column=0, sticky="nsew")

        # general tab
        general_frame = ctk.CTkScrollableFrame(tabview.add("general"), fg_color="transparent")
        general_frame.grid_columnconfigure(1, weight=1)
        general_frame.grid_columnconfigure(2, weight=1)
        self.build_general_tab(general_frame, controller, ui_state, text_ui_state)
        general_frame.pack(fill="both", expand=1)

        # image augmentation tab
        image_aug_master = tabview.add("image augmentation")
        image_aug_frame = ctk.CTkScrollableFrame(image_aug_master, fg_color="transparent")
        image_aug_frame.grid_columnconfigure(0, weight=0)
        image_aug_frame.grid_columnconfigure(1, weight=0)
        image_aug_frame.grid_columnconfigure(2, weight=0)
        image_aug_frame.grid_columnconfigure(3, weight=1)
        self.build_image_augmentation_tab(image_aug_frame, controller, image_ui_state)

        # image
        image_preview, filename_preview, caption_preview = controller.get_preview_image(self.image_preview_file_index, self.preview_augmentations.get())
        self.image = ctk.CTkImage(
            light_image=image_preview,
            size=image_preview.size,
        )
        image_label = ctk.CTkLabel(master=image_aug_frame, text="", image=self.image, height=300, width=300)
        image_label.grid(row=0, column=4, rowspan=6)

        # refresh preview
        update_button_frame = ctk.CTkFrame(master=image_aug_frame, corner_radius=0, fg_color="transparent")
        update_button_frame.grid(row=6, column=4, rowspan=6, sticky="nsew")
        update_button_frame.grid_columnconfigure(1, weight=1)

        prev_preview_button = self.components.button(update_button_frame, 0, 0, "<", command=self._prev_image_preview)
        self.components.button(update_button_frame, 0, 1, "Update Preview", command=self._update_image_preview)
        next_preview_button = self.components.button(update_button_frame, 0, 2, ">", command=self._next_image_preview)
        preview_augmentations_switch = ctk.CTkSwitch(update_button_frame, text="Show Augmentations", variable=self.preview_augmentations, command=self._update_image_preview)
        preview_augmentations_switch.grid(row=1, column=0, columnspan=3, padx=5, pady=5)

        prev_preview_button.configure(width=40)
        next_preview_button.configure(width=40)

        #caption and filename preview
        self.filename_preview = ctk.CTkLabel(master=update_button_frame, text=filename_preview, width=300, anchor="nw", justify="left", padx=10, wraplength=280)
        self.filename_preview.grid(row=2, column=0, columnspan=3)
        self.caption_preview = ctk.CTkTextbox(master=update_button_frame, width = 300, height = 150, wrap="word", border_width=2)
        self.caption_preview.insert(index="1.0", text=caption_preview)
        self.caption_preview.configure(state="disabled")
        self.caption_preview.grid(row=3, column=0, columnspan=3, rowspan=3)

        image_aug_frame.pack(fill="both", expand=1)

        # text augmentation tab
        text_aug_frame = ctk.CTkScrollableFrame(tabview.add("text augmentation"), fg_color="transparent")
        text_aug_frame.grid_columnconfigure(0, weight=0)
        text_aug_frame.grid_columnconfigure(1, weight=0)
        text_aug_frame.grid_columnconfigure(2, weight=0)
        text_aug_frame.grid_columnconfigure(3, weight=1)
        self.build_text_augmentation_tab(text_aug_frame, controller, text_ui_state)
        text_aug_frame.pack(fill="both", expand=1)

        # statistics tab
        stats_frame = ctk.CTkScrollableFrame(tabview.add("statistics"), fg_color="transparent")
        stats_frame.grid_columnconfigure(0, weight=0, minsize=150)
        stats_frame.grid_columnconfigure(1, weight=0, minsize=150)
        stats_frame.grid_columnconfigure(2, weight=0, minsize=150)
        stats_frame.grid_columnconfigure(3, weight=0, minsize=150)
        self.build_concept_stats_tab(stats_frame, controller)

        #aspect bucketing plot, mostly copied from timestep preview graph
        appearance_mode = AppearanceModeTracker.get_mode()
        background_color = self.winfo_rgb(ThemeManager.theme["CTkToplevel"]["fg_color"][appearance_mode])
        text_color = self.winfo_rgb(ThemeManager.theme["CTkLabel"]["text_color"][appearance_mode])
        background_color = f"#{int(background_color[0]/256):x}{int(background_color[1]/256):x}{int(background_color[2]/256):x}"
        self.text_color = f"#{int(text_color[0]/256):x}{int(text_color[1]/256):x}{int(text_color[2]/256):x}"

        plt.set_loglevel('WARNING')     #suppress errors about data type in bar chart

        assert self.bucket_fig is None
        self.bucket_fig, self.bucket_ax = plt.subplots(figsize=(7,3))
        self.canvas = FigureCanvasTkAgg(self.bucket_fig, master=stats_frame)
        self.canvas.get_tk_widget().grid(row=19, column=0, columnspan=4, rowspan=2)
        self.bucket_fig.tight_layout()
        self.bucket_fig.subplots_adjust(bottom=0.15)

        self.bucket_fig.set_facecolor(background_color)
        self.bucket_ax.set_facecolor(background_color)
        self.bucket_ax.spines['bottom'].set_color(self.text_color)
        self.bucket_ax.spines['left'].set_color(self.text_color)
        self.bucket_ax.spines['top'].set_visible(False)
        self.bucket_ax.spines['right'].set_color(self.text_color)
        self.bucket_ax.tick_params(axis='x', colors=self.text_color, which="both")
        self.bucket_ax.tick_params(axis='y', colors=self.text_color, which="both")
        self.bucket_ax.xaxis.label.set_color(self.text_color)
        self.bucket_ax.yaxis.label.set_color(self.text_color)

        stats_frame.pack(fill="both", expand=1)

        #automatic concept scan
        self.scan_thread = threading.Thread(target=controller.auto_update_concept_stats, args=[self], daemon=True)
        self.scan_thread.start()

        self.components.button(self, 1, 0, "ok", self._ok)

        self.wait_visibility()
        self.grab_set()
        self.focus_set()
        self.after(200, lambda: set_window_icon(self))

    def _prev_image_preview(self):
        self.image_preview_file_index = max(self.image_preview_file_index - 1, 0)
        self._update_image_preview()

    def _next_image_preview(self):
        self.image_preview_file_index += 1
        self._update_image_preview()

    def _update_image_preview(self):
        image_preview, filename_preview, caption_preview = self.controller.get_preview_image(self.image_preview_file_index, self.preview_augmentations.get())
        self.image.configure(light_image=image_preview, size=image_preview.size)
        self.filename_preview.configure(text=filename_preview)
        self.caption_preview.configure(state="normal")
        self.caption_preview.delete(index1="1.0", index2="end")
        self.caption_preview.insert(index="1.0", text=caption_preview)
        self.caption_preview.configure(state="disabled")

    def destroy(self):
        if self.bucket_fig is not None:
            plt.close(self.bucket_fig)
            self.bucket_fig = None

        super().destroy()

    def _ok(self):
        self.destroy()
