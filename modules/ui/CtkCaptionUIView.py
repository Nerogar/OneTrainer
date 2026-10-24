from tkinter import filedialog

from modules.ui.BaseCaptionUIView import BaseCaptionUIView
from modules.ui.CaptionUIController import CaptionUIController
from modules.ui.CtkGenerateCaptionsWindowView import CtkGenerateCaptionsWindowView
from modules.ui.CtkGenerateMasksWindowView import CtkGenerateMasksWindowView
from modules.util.ui import ctk_components
from modules.util.ui.CtkUIState import CtkUIState
from modules.util.ui.ui_utils import bind_mousewheel, set_window_icon

import customtkinter as ctk
from customtkinter import ScalingTracker, ThemeManager
from PIL import Image


class CtkCaptionUIView(BaseCaptionUIView, ctk.CTkToplevel):
    def __init__(self, parent, controller: CaptionUIController, *args, **kwargs):
        ctk.CTkToplevel.__init__(self, parent, *args, **kwargs)
        BaseCaptionUIView.__init__(self, ctk_components)
        self.protocol("WM_DELETE_WINDOW", controller.on_close)

        self.controller = controller
        controller.view = self
        self.config_ui_state = CtkUIState(self, controller.config_ui_data)
        self.enable_mask_editing_var = ctk.BooleanVar()
        self.mask_editing_alpha = None
        self.prompt_var = None
        self.prompt_component = None
        self.image = None
        self.image_label = None
        self.file_list = None
        self.image_labels = []

        self.title("OneTrainer")
        self.geometry("1280x980")
        self.resizable(False, False)

        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        top_frame = ctk.CTkFrame(self)
        top_frame.grid(row=0, column=0, sticky="nsew")
        self.build_top_bar(top_frame, controller, self.config_ui_state)

        self.bottom_frame = ctk.CTkFrame(self)
        self.bottom_frame.grid(row=1, column=0, sticky="nsew")
        self.bottom_frame.grid_rowconfigure(0, weight=1)
        self.bottom_frame.grid_columnconfigure(0, weight=0)
        self.bottom_frame.grid_columnconfigure(1, weight=1)

        self.file_list_column(self.bottom_frame)
        self.content_column(self.bottom_frame)
        self.controller.load_directory()

        self.wait_visibility()
        self.focus_set()
        self.after(200, lambda: set_window_icon(self))

    def file_list_column(self, master):
        if self.file_list is not None:
            self.image_labels = []
            self.file_list.destroy()

        self.file_list = ctk.CTkScrollableFrame(master, width=300)
        self.file_list.grid(row=0, column=0, sticky="nsew")

        for i, filename in enumerate(self.controller.image_rel_paths):
            def __create_switch_image(index):
                def __switch_image(event):
                    self.controller.switch_image(index)

                return __switch_image

            label = ctk.CTkLabel(self.file_list, text=filename)
            label.bind("<Button-1>", __create_switch_image(i))

            self.image_labels.append(label)
            label.grid(row=i, column=0, padx=5, sticky="nsw")

    def content_column(self, master):
        image = Image.new("RGBA", (512, 512), (0, 0, 0, 0))

        right_frame = ctk.CTkFrame(master, fg_color="transparent")
        right_frame.grid(row=0, column=1, sticky="nsew")

        right_frame.grid_columnconfigure(4, weight=1)
        right_frame.grid_rowconfigure(1, weight=1)

        self.build_mask_buttons(right_frame)

        # checkbox to enable mask editing
        self.enable_mask_editing_var = ctk.BooleanVar()
        self.enable_mask_editing_var.set(False)
        enable_mask_editing_checkbox = ctk.CTkCheckBox(
            right_frame, text="Enable Mask Editing", variable=self.enable_mask_editing_var, width=50)
        enable_mask_editing_checkbox.grid(row=0, column=2, padx=25, pady=5, sticky="w")

        # mask alpha textbox
        self.mask_editing_alpha = ctk.CTkEntry(master=right_frame, width=40, placeholder_text="1.0")
        self.mask_editing_alpha.insert(0, "1.0")
        self.mask_editing_alpha.grid(row=0, column=3, sticky="e", padx=5, pady=5)
        self.bind_key_events(self.mask_editing_alpha)

        mask_editing_alpha_label = ctk.CTkLabel(right_frame, text="Brush Alpha", width=75)
        mask_editing_alpha_label.grid(row=0, column=4, padx=0, pady=5, sticky="w")

        # image
        self.image = ctk.CTkImage(
            light_image=image,
            size=(self.controller.image_size, self.controller.image_size)
        )
        self.image_label = ctk.CTkLabel(
            master=right_frame, text="", image=self.image,
            height=self.controller.image_size, width=self.controller.image_size
        )
        self.image_label.grid(row=1, column=0, columnspan=5, sticky="nsew")

        self.image_label.bind("<Motion>", self.edit_mask)
        self.image_label.bind("<Button-1>", self.edit_mask)
        self.image_label.bind("<Button-3>", self.edit_mask)
        bind_mousewheel(self.image_label, {self.image_label.children["!label"]}, self.draw_mask_radius)

        # prompt
        self.prompt_var = ctk.StringVar()
        self.prompt_component = ctk.CTkEntry(right_frame, textvariable=self.prompt_var)
        self.prompt_component.grid(row=2, column=0, columnspan=5, pady=5, sticky="new")
        self.bind_key_events(self.prompt_component)
        self.prompt_component.focus_set()

    def bind_key_events(self, component):
        component.bind("<Down>", lambda e: self.controller.next_image())
        component.bind("<Up>", lambda e: self.controller.previous_image())
        component.bind("<Return>", self.save)
        component.bind("<Control-m>", self.toggle_mask)
        component.bind("<Control-d>", self.draw_mask_editing_mode)
        component.bind("<Control-f>", self.fill_mask_editing_mode)

    def refresh_file_list(self):
        self.file_list_column(self.bottom_frame)

    def focus_prompt(self):
        self.prompt_component.focus_set()

    def on_image_switched(self, old_index, new_index, prompt):
        if len(self.image_labels) > 0 and old_index < len(self.image_labels):
            self.image_labels[old_index].configure(
                text_color=ThemeManager.theme["CTkLabel"]["text_color"])
        self.image_labels[new_index].configure(text_color="#FF0000")
        self.refresh_image()
        self.prompt_var.set(prompt)

    def on_image_cleared(self):
        image = Image.new("RGB", (512, 512), (0, 0, 0))
        self.image.configure(light_image=image)

    def refresh_image(self):
        pil_image, size = self.controller.get_display_image()
        self.image.configure(light_image=pil_image, size=size)

    def draw_mask_radius(self, delta, raw_event):
        self.controller.update_mask_draw_radius(delta)

    def edit_mask(self, event):
        if not self.enable_mask_editing_var.get():
            return

        if event.widget != self.image_label.children["!label"]:
            return

        display_scaling = ScalingTracker.get_window_scaling(self)

        event_x = event.x / display_scaling
        event_y = event.y / display_scaling

        is_right = False
        is_left = False
        if event.state & 0x0100 or event.num == 1:  # left mouse button
            is_left = True
        elif event.state & 0x0400 or event.num == 3:  # right mouse button
            is_right = True

        try:
            alpha = float(self.mask_editing_alpha.get())
        except Exception:
            alpha = 1.0

        self.controller.handle_edit_mask(event_x, event_y, is_left, is_right, alpha)

    def save(self, event):
        self.controller.save(self.prompt_var.get())

    def draw_mask_editing_mode(self, *args):
        self.controller.set_mask_editing_mode('draw')

        if args:
            # disable default event
            return "break"
        return None

    def fill_mask_editing_mode(self, *args):
        self.controller.set_mask_editing_mode('fill')

    def toggle_mask(self, *args):
        self.controller.toggle_mask()
        self.refresh_image()

    def open_directory(self):
        new_dir = filedialog.askdirectory()

        if new_dir:
            self.controller.dir = new_dir
            self.controller.load_directory(include_subdirectories=self.controller.config_ui_data["include_subdirectories"])

    def open_mask_window(self):
        self.wait_window(self.controller.open_mask_window(self, CtkGenerateMasksWindowView))
        self.controller.switch_image(self.controller.current_image_index)

    def open_caption_window(self):
        self.wait_window(self.controller.open_caption_window(self, CtkGenerateCaptionsWindowView))
        self.controller.switch_image(self.controller.current_image_index)

    def open_in_explorer(self):
        self.controller.open_in_explorer()

    def destroy(self):
        self.controller._release_models()
        super().destroy()
