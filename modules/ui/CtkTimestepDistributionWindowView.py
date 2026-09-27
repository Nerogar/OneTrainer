
from modules.ui.BaseTimestepDistributionWindowView import BaseTimestepDistributionWindowView
from modules.ui.TimestepDistributionWindowController import TimestepDistributionWindowController
from modules.util.ui import ctk_components
from modules.util.ui.ui_utils import set_window_icon

import customtkinter as ctk
from customtkinter import AppearanceModeTracker, ThemeManager
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class CtkTimestepDistributionWindowView(BaseTimestepDistributionWindowView, ctk.CTkToplevel):
    def __init__(
            self,
            parent,
            controller: TimestepDistributionWindowController,
            ui_state,
            *args, **kwargs,
    ):
        ctk.CTkToplevel.__init__(self, parent, *args, **kwargs)
        BaseTimestepDistributionWindowView.__init__(self, ctk_components)

        self.title("Timestep Distribution")
        self.geometry("900x600")
        self.resizable(True, True)

        self.controller = controller
        self.ax = None
        self.canvas = None

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        frame = ctk.CTkScrollableFrame(self, fg_color="transparent")
        frame.grid_columnconfigure(0, weight=0)
        frame.grid_columnconfigure(1, weight=0)
        frame.grid_columnconfigure(2, weight=0)
        frame.grid_columnconfigure(3, weight=1)
        frame.grid_rowconfigure(7, weight=1)

        self.build_content(frame, controller, ui_state)

        # matplotlib chart (CTK-only: needs winfo_rgb from the toplevel)
        appearance_mode = AppearanceModeTracker.get_mode()
        background_color = self.winfo_rgb(ThemeManager.theme["CTkToplevel"]["fg_color"][appearance_mode])
        text_color = self.winfo_rgb(ThemeManager.theme["CTkLabel"]["text_color"][appearance_mode])
        background_color = f"#{int(background_color[0]/256):x}{int(background_color[1]/256):x}{int(background_color[2]/256):x}"
        text_color = f"#{int(text_color[0]/256):x}{int(text_color[1]/256):x}{int(text_color[2]/256):x}"

        fig, ax = plt.subplots()
        self.ax = ax
        self.canvas = FigureCanvasTkAgg(fig, master=frame)
        self.canvas.get_tk_widget().grid(row=0, column=3, rowspan=8)

        fig.set_facecolor(background_color)
        ax.set_facecolor(background_color)
        ax.spines['bottom'].set_color(text_color)
        ax.spines['left'].set_color(text_color)
        ax.spines['top'].set_color(text_color)
        ax.spines['right'].set_color(text_color)
        ax.tick_params(axis='x', colors=text_color, which="both")
        ax.tick_params(axis='y', colors=text_color, which="both")
        ax.xaxis.label.set_color(text_color)
        ax.yaxis.label.set_color(text_color)

        self.__update_preview()

        # update button
        ctk_components.button(frame, 8, 3, "Update Preview", command=self.__update_preview)

        frame.pack(fill="both", expand=1)
        frame.grid(row=0, column=0, sticky='nsew')
        ctk_components.button(self, 1, 0, "ok", self.destroy)

        self.wait_visibility()
        self.after(200, lambda: set_window_icon(self))
        self.grab_set()
        self.focus_set()

    def __update_preview(self):
        self.ax.cla()
        self.ax.hist(self.controller.generate_preview_data(), bins=1000, range=(0, 999))
        self.canvas.draw()
