
from modules.modelSetup.mixin.ModelSetupNoiseMixin import ModelSetupNoiseMixin

from modules.util.config.ConceptConfig import ConceptNoiseConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.TimestepDistribution import TimestepDistribution
from modules.util.ui import components
from modules.util.ui.ui_utils import set_window_icon
from modules.util.ui.UIState import UIState

import torch
from torch import Tensor

import customtkinter as ctk
from customtkinter import AppearanceModeTracker, ThemeManager
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class TimestepGenerator(ModelSetupNoiseMixin):

    def __init__(
            self,
            timestep_distribution: TimestepDistribution,
            min_noising_strength: float,
            max_noising_strength: float,
            noising_weight: float,
            noising_bias: float,
            timestep_shift: float,
    ):
        super().__init__()

        self.timestep_distribution = timestep_distribution
        self.min_noising_strength = min_noising_strength
        self.max_noising_strength = max_noising_strength
        self.noising_weight = noising_weight
        self.noising_bias = noising_bias
        self.timestep_shift = timestep_shift

    def generate(self) -> Tensor:
        generator = torch.Generator()
        generator.seed()

        config = TrainConfig.default_values()
        config.timestep_distribution = self.timestep_distribution
        config.min_noising_strength = self.min_noising_strength
        config.max_noising_strength = self.max_noising_strength
        config.noising_weight = self.noising_weight
        config.noising_bias = self.noising_bias
        config.timestep_shift = self.timestep_shift


        return self._get_timestep_discrete(
            num_train_timesteps=1000,
            deterministic=False,
            generator=generator,
            batch_size=1000000,
            config=config,
        )


class TimestepDistributionFrame(ctk.CTkFrame):
    def __init__(self,
            parent,
            config: TrainConfig | ConceptNoiseConfig,
            ui_state: UIState,
            *args, **kwargs
    ):
        super().__init__(parent, *args, fg_color="transparent", **kwargs)

        self.train_config = config
        self.ui_state = ui_state
        self.ax = None
        self.canvas = None

        self.__create_frame()

    def __create_frame(self):
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=0)
        self.grid_columnconfigure(2, weight=0)
        self.grid_columnconfigure(3, weight=1)
        self.grid_rowconfigure(8, weight=1)

        # timestep distribution
        components.label(self, 0, 0, "Timestep Distribution",
                         tooltip="Selects the function to sample timesteps during training",
                         wide_tooltip=True)
        components.options(self, 0, 1, [str(x) for x in list(TimestepDistribution)], self.ui_state,
                           "timestep_distribution")

        # min noising strength
        components.label(self, 1, 0, "Min Noising Strength",
                         tooltip="Specifies the minimum noising strength used during training. This can help to improve composition, but prevents finer details from being trained")
        components.entry(self, 1, 1, self.ui_state, "min_noising_strength")

        # max noising strength
        components.label(self, 2, 0, "Max Noising Strength",
                         tooltip="Specifies the maximum noising strength used during training. This can be useful to reduce overfitting, but also reduces the impact of training samples on the overall image composition")
        components.entry(self, 2, 1, self.ui_state, "max_noising_strength")

        # noising weight
        components.label(self, 3, 0, "Noising Weight",
                         tooltip="Controls the weight parameter of the timestep distribution function. Use the preview to see more details.")
        components.entry(self, 3, 1, self.ui_state, "noising_weight")

        # noising bias
        components.label(self, 4, 0, "Noising Bias",
                         tooltip="Controls the bias parameter of the timestep distribution function. Use the preview to see more details.")
        components.entry(self, 4, 1, self.ui_state, "noising_bias")

        # timestep shift
        components.label(self, 5, 0, "Timestep Shift",
                         tooltip="Shift the timestep distribution. Use the preview to see more details.")
        components.entry(self, 5, 1, self.ui_state, "timestep_shift")

        # dynamic timestep shifting
        components.label(self, 6, 0, "Dynamic Timestep Shifting",
                         tooltip="Dynamically shift the timestep distribution based on resolution. If enabled, the shifting parameters are taken from the model's scheduler configuration and Timestep Shift is ignored. Dynamic Timestep Shifting is not shown in the preview.")
        components.switch(self, 6, 1, self.ui_state, "dynamic_timestep_shifting")

        # update button
        update_button = components.button(self, 7, 0, "Update Preview", command=self.update_preview)
        update_button.grid(columnspan=2)

        # plot
        appearance_mode = AppearanceModeTracker.get_mode()
        background_color = self.winfo_rgb(ThemeManager.theme["CTkToplevel"]["fg_color"][appearance_mode])
        text_color = self.winfo_rgb(ThemeManager.theme["CTkLabel"]["text_color"][appearance_mode])
        background_color = f"#{int(background_color[0]/256):x}{int(background_color[1]/256):x}{int(background_color[2]/256):x}"
        text_color = f"#{int(text_color[0]/256):x}{int(text_color[1]/256):x}{int(text_color[2]/256):x}"

        fig, ax = plt.subplots()
        self.ax = ax
        self.canvas = FigureCanvasTkAgg(fig, master=self)
        self.canvas.get_tk_widget().grid(row=0, column=3, rowspan=9)

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

    def update_preview(self):
        generator = TimestepGenerator(
            timestep_distribution=self.train_config.timestep_distribution,
            min_noising_strength=self.train_config.min_noising_strength,
            max_noising_strength=self.train_config.max_noising_strength,
            noising_weight=self.train_config.noising_weight,
            noising_bias=self.train_config.noising_bias,
            timestep_shift=self.train_config.timestep_shift,
        )

        self.ax.cla()
        self.ax.hist(generator.generate(), bins=1000, range=(0, 999))
        self.canvas.draw()


class TimestepDistributionWindow(ctk.CTkToplevel):
    def __init__(
            self,
            parent,
            config: TrainConfig,
            ui_state: UIState,
            *args, **kwargs,
    ):
        super().__init__(parent, *args, **kwargs)

        self.title("Timestep Distribution")
        self.geometry("900x600")
        self.resizable(True, True)

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        frame = TimestepDistributionFrame(self, config, ui_state)
        frame.grid(row=0, column=0, sticky='nsew')

        components.button(self, 1, 0, "ok", self.__ok)

        self.wait_visibility()
        self.after(200, lambda: set_window_icon(self))
        self.grab_set()
        self.focus_set()

        frame.update_preview()

    def __ok(self):
        self.destroy()
