from modules.modelSetup.mixin.ModelSetupNoiseMixin import ModelSetupNoiseMixin
from modules.util.config.ConceptConfig import ConceptOverridesConfig
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

        batch_config = {
            "min_noising_strength": self.min_noising_strength,
            "max_noising_strength": self.max_noising_strength,
            "noising_weight": self.noising_weight,
            "noising_bias": self.noising_bias,
            "timestep_shift": self.timestep_shift,
        }
        batch_config = {k: torch.tensor(v).unsqueeze(0) for k, v in batch_config.items()}

        return self._get_timestep_discrete(
            num_train_timesteps=1000,
            deterministic=False,
            generator=generator,
            batch_size=1000000,
            config=config,
            batch_config=batch_config,
        )


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

        self.config = config
        self.ui_state = ui_state

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        frame = self.__content_frame(self)
        frame.grid(row=0, column=0, sticky='nsew')
        components.button(self, 1, 0, "ok", self.__ok)

        self.wait_visibility()
        self.after(200, lambda: set_window_icon(self))
        self.grab_set()
        self.focus_set()

    def __content_frame(self, master):
        frame = ctk.CTkScrollableFrame(master, fg_color="transparent")
        frame.grid_columnconfigure(0, weight=0)
        frame.grid_columnconfigure(1, weight=0)
        frame.grid_columnconfigure(2, weight=1)
        frame.grid_rowconfigure(7, weight=1)

        # timestep distribution
        components.label(frame, 0, 0, "Timestep Distribution",
                         tooltip="Selects the function to sample timesteps during training",
                         wide_tooltip=True)
        components.options(frame, 0, 1, [str(x) for x in list(TimestepDistribution)], self.ui_state,
                           "timestep_distribution")

        # min noising strength
        components.label(frame, 1, 0, "Min Noising Strength",
                         tooltip="Specifies the minimum noising strength used during training. This can help to improve composition, but prevents finer details from being trained")
        components.entry(frame, 1, 1, self.ui_state, "min_noising_strength")

        # max noising strength
        components.label(frame, 2, 0, "Max Noising Strength",
                         tooltip="Specifies the maximum noising strength used during training. This can be useful to reduce overfitting, but also reduces the impact of training samples on the overall image composition")
        components.entry(frame, 2, 1, self.ui_state, "max_noising_strength")

        # noising weight
        components.label(frame, 3, 0, "Noising Weight",
                         tooltip="Controls the weight parameter of the timestep distribution function. Use the preview to see more details.")
        components.entry(frame, 3, 1, self.ui_state, "noising_weight")

        # noising bias
        components.label(frame, 4, 0, "Noising Bias",
                         tooltip="Controls the bias parameter of the timestep distribution function. Use the preview to see more details.")
        components.entry(frame, 4, 1, self.ui_state, "noising_bias")

        # timestep shift
        components.label(frame, 5, 0, "Timestep Shift",
                         tooltip="Shift the timestep distribution. Use the preview to see more details.")
        components.entry(frame, 5, 1, self.ui_state, "timestep_shift")

        # dynamic timestep shifting
        components.label(frame, 6, 0, "Dynamic Timestep Shifting",
                         tooltip="Dynamically shift the timestep distribution based on resolution. If enabled, the shifting parameters are taken from the model's scheduler configuration and Timestep Shift is ignored. Dynamic Timestep Shifting is not shown in the preview.")
        components.switch(frame, 6, 1, self.ui_state, "dynamic_timestep_shifting")

        # plot
        plot = TimestepDistributionPlot(frame, self.config)
        plot.get_tk_widget().grid(row=0, column=2, rowspan=8)
        plot.update_preview()

        # update button
        update_button = components.button(frame, 7, 0, "Update Preview", command=plot.update_preview)
        update_button.grid(columnspan=2)

        frame.pack(fill="both", expand=1)
        return frame

    def __ok(self):
        self.destroy()


class TimestepDistributionPlot(FigureCanvasTkAgg):
    def __init__(
            self,
            parent,
            config: TrainConfig,
            config_overrides: ConceptOverridesConfig | None = None
    ):
        fig, ax = plt.subplots()

        super().__init__(fig, parent)
        self.ax = ax
        self._config = config
        self._config_overrides = config_overrides

        appearance_mode = AppearanceModeTracker.get_mode()
        background_color = parent.winfo_rgb(ThemeManager.theme["CTkToplevel"]["fg_color"][appearance_mode])
        text_color = parent.winfo_rgb(ThemeManager.theme["CTkLabel"]["text_color"][appearance_mode])
        background_color = f"#{int(background_color[0]/256):x}{int(background_color[1]/256):x}{int(background_color[2]/256):x}"
        text_color = f"#{int(text_color[0]/256):x}{int(text_color[1]/256):x}{int(text_color[2]/256):x}"

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
            timestep_distribution=self._config.timestep_distribution,
            min_noising_strength=self.__get_config_value('min_noising_strength'),
            max_noising_strength=self.__get_config_value('max_noising_strength'),
            noising_weight=self.__get_config_value('noising_weight'),
            noising_bias=self.__get_config_value('noising_bias'),
            timestep_shift=self.__get_config_value('timestep_shift'),
        )

        self.ax.cla()
        self.ax.hist(generator.generate(), bins=1000, range=(0, 999))
        self.draw()

    def __get_config_value(self, attr: str):
        if self._config_overrides is not None:
            value = getattr(self._config_overrides, attr)
            if value is not None:
                return value
        return getattr(self._config, attr)
