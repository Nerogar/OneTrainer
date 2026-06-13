import contextlib
import tkinter as tk
import traceback

from modules.modelSampler.BaseModelSampler import (
    ModelSamplerOutput,
)
from modules.ui.BaseSampleWindowView import BaseSampleWindowView
from modules.ui.CtkSampleFrameView import CtkSampleFrameView
from modules.ui.SampleFrameController import SampleFrameController
from modules.ui.SampleWindowController import SampleWindowController
from modules.util.enum.FileType import FileType
from modules.util.ui import ctk_components
from modules.util.ui.CtkUIState import CtkUIState
from modules.util.ui.ui_utils import set_window_icon

import customtkinter as ctk
from PIL import Image


class CtkSampleWindowView(BaseSampleWindowView, ctk.CTkToplevel):
    def __init__(
            self,
            parent,
            controller: SampleWindowController,
            *args, **kwargs
    ):
        ctk.CTkToplevel.__init__(self, parent, *args, **kwargs)
        BaseSampleWindowView.__init__(self, ctk_components)

        self.title("Sample")
        self.geometry("1200x800")
        self.resizable(True, True)

        model_type = controller.get_model_type()
        self.ui_state = CtkUIState(self, controller.sample)

        if controller.use_external_model:
            controller.callbacks.set_on_sample_custom(self.__update_preview)
            controller.callbacks.set_on_update_sample_custom_progress(self.__update_progress)

        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=0)
        self.grid_rowconfigure(3, weight=0)
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)

        prompt_frame = CtkSampleFrameView(self, SampleFrameController(controller.sample, model_type), self.ui_state, include_settings=False)
        prompt_frame.grid(row=0, column=0, columnspan=2, padx=0, pady=0, sticky="nsew")

        settings_frame = CtkSampleFrameView(self, SampleFrameController(controller.sample, model_type), self.ui_state, include_prompt=False)
        settings_frame.grid(row=1, column=0, padx=0, pady=0, sticky="nsew")

        # image
        self.image = ctk.CTkImage(
            light_image=self.__dummy_image(),
            size=(512, 512)
        )

        image_label = ctk.CTkLabel(master=self, text="", image=self.image, height=512, width=512)
        image_label.grid(row=1, column=1, rowspan=3, sticky="nsew")

        self.progress = self.components.progress(self, 2, 0)
        self.components.button(self, 3, 0, "sample",
                               lambda: controller.do_sample(self.__update_preview, self.__update_progress))

        self.wait_visibility()
        self.focus_set()
        self.after(200, lambda: set_window_icon(self))

    def __update_preview(self, sampler_output: ModelSamplerOutput):
        if sampler_output.file_type == FileType.IMAGE:
            image = sampler_output.data
            self.image.configure(
                light_image=image,
                size=(image.width, image.height),
            )

    def __update_progress(self, progress: int, max_progress: int):
        self.progress.set(progress / max_progress)
        self.update()

    def __dummy_image(self) -> Image:
        return Image.new(mode="RGB", size=(512, 512), color=(0, 0, 0))

    def destroy(self):
        try:
            if hasattr(self, "_icon_image_ref"):
                del self._icon_image_ref

            # Remove any pending after callbacks
            for after_id in self.tk.call('after', 'info'):
                with contextlib.suppress(tk.TclError, RuntimeError):
                    self.after_cancel(after_id)

            super().destroy()
        except (tk.TclError, RuntimeError) as e:
            print(f"Error destroying window: {e}")
        except Exception as e:
            print(f"Unexpected error destroying window: {e}")
            traceback.print_exc()
