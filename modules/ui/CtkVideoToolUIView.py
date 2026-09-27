from modules.ui.BaseVideoToolUIView import BaseVideoToolUIView
from modules.ui.VideoToolUIController import VideoToolUIController
from modules.util.image_util import load_image
from modules.util.ui import ctk_components
from modules.util.ui.CtkUIState import CtkUIState

import customtkinter as ctk

PAD = ctk_components.PAD


class CtkVideoToolUIView(BaseVideoToolUIView, ctk.CTkToplevel):
    def __init__(self, parent, controller: VideoToolUIController, *args, **kwargs):
        ctk.CTkToplevel.__init__(self, parent, *args, **kwargs)
        BaseVideoToolUIView.__init__(self, ctk_components)

        self.controller = controller
        ui_state = CtkUIState(self, controller.args)

        self.title("Video Tools")
        self.geometry("600x720")
        self.resizable(True, True)
        self.wait_visibility()
        self.focus_set()

        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)
        self.grid_columnconfigure(0, weight=1)

        tabview = ctk.CTkTabview(self)
        tabview.grid(row=0, column=0, sticky="nsew")

        clip_frame = ctk.CTkScrollableFrame(tabview.add("extract clips"), fg_color="transparent")
        clip_frame.grid_columnconfigure(0, weight=0, minsize=120)
        clip_frame.grid_columnconfigure(1, weight=0, minsize=200)
        clip_frame.grid_columnconfigure(2, weight=0)
        clip_frame.grid_columnconfigure(3, weight=1)
        self.build_clip_extract_tab(clip_frame, controller, ui_state)
        clip_frame.pack(fill="both", expand=1)

        image_frame = ctk.CTkScrollableFrame(tabview.add("extract images"), fg_color="transparent")
        image_frame.grid_columnconfigure(0, weight=0, minsize=120)
        image_frame.grid_columnconfigure(1, weight=0, minsize=200)
        image_frame.grid_columnconfigure(2, weight=0)
        image_frame.grid_columnconfigure(3, weight=1)
        self.build_image_extract_tab(image_frame, controller, ui_state)
        image_frame.pack(fill="both", expand=1)

        download_frame = ctk.CTkScrollableFrame(tabview.add("download"), fg_color="transparent")
        download_frame.grid_columnconfigure(0, weight=0, minsize=120)
        download_frame.grid_columnconfigure(1, weight=0, minsize=200)
        download_frame.grid_columnconfigure(2, weight=0)
        download_frame.grid_columnconfigure(3, weight=1)
        self.build_video_download_tab(download_frame, controller, ui_state)
        download_frame.pack(fill="both", expand=1)

        self._build_status_bar(self)

    def _build_status_bar(self, master):
        frame = ctk.CTkFrame(master, fg_color="transparent")
        frame.grid(row=1, column=0)
        frame.grid_columnconfigure(0, weight=0, minsize=160)
        frame.grid_columnconfigure(1, weight=0, minsize=300)
        frame.grid_columnconfigure(2, weight=1)

        preview_path = "resources/icons/icon.png"
        preview = load_image(preview_path, 'RGB')
        preview.thumbnail((150, 150))
        self.preview_image = ctk.CTkImage(light_image=preview, size=preview.size)
        self.preview_image_label = ctk.CTkLabel(
            master=frame, text="Preview image", image=self.preview_image, height=150, width=150,
            compound="top")
        self.preview_image_label.grid(row=0, column=0, sticky="nw", padx=5, pady=5)

        self.status_label = ctk.CTkTextbox(master=frame, width=400, height=160, wrap="word", border_width=2)
        self.status_label.insert(index="1.0", text="Current status")
        self.status_label.configure(state="disabled")
        self.status_label.grid(row=0, column=1, sticky="ne", padx=5, pady=5)

    def _create_textbox(self, master, row, col, width, height, ui_state, var_name):
        var = ui_state.get_var(var_name)
        textbox = ctk.CTkTextbox(master, width=width, height=height, border_width=2)
        textbox.insert("1.0", var.get())
        textbox.grid(row=row, column=col, rowspan=2, sticky="w", padx=PAD, pady=PAD)

        def on_text_change(event=None):
            var.set(textbox.get("1.0", "end-1c"))

        textbox.bind("<KeyRelease>", on_text_change)
        return textbox

    def update_status(self, status_text: str):
        self.status_label.configure(state="normal")
        self.status_label.insert(index="end", text=status_text + "\n")
        self.status_label.configure(state="disabled")

    def clear_status(self):
        self.status_label.configure(state="normal")
        self.status_label.delete(index1="1.0", index2="end")
        self.status_label.configure(state="disabled")

    def update_preview(self, preview_image, label_text: str):
        self.preview_image.configure(light_image=preview_image, size=preview_image.size)
        self.preview_image_label.configure(text=label_text)
