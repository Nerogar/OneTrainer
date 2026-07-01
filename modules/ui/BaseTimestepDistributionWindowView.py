




class BaseTimestepDistributionWindowView:
    def __init__(self, components):
        self.components = components

    def build_content(self, frame, controller, ui_state):
        # timestep distribution
        self.components.label(frame, 0, 0, "Timestep Distribution",
                         tooltip="Selects the function to sample timesteps during training",
                         wide_tooltip=True)
        self.components.options(frame, 0, 1, controller.get_distribution_options(), ui_state,
                           "timestep_distribution")

        # min noising strength
        self.components.label(frame, 1, 0, "Min Noising Strength",
                         tooltip="Specifies the minimum noising strength used during training. This can help to improve composition, but prevents finer details from being trained")
        self.components.entry(frame, 1, 1, ui_state, "min_noising_strength")

        # max noising strength
        self.components.label(frame, 2, 0, "Max Noising Strength",
                         tooltip="Specifies the maximum noising strength used during training. This can be useful to reduce overfitting, but also reduces the impact of training samples on the overall image composition")
        self.components.entry(frame, 2, 1, ui_state, "max_noising_strength")

        # noising weight
        self.components.label(frame, 3, 0, "Noising Weight",
                         tooltip="Controls the weight parameter of the timestep distribution function. Use the preview to see more details.")
        self.components.entry(frame, 3, 1, ui_state, "noising_weight")

        # noising bias
        self.components.label(frame, 4, 0, "Noising Bias",
                         tooltip="Controls the bias parameter of the timestep distribution function. Use the preview to see more details.")
        self.components.entry(frame, 4, 1, ui_state, "noising_bias")

        # timestep shift
        self.components.label(frame, 5, 0, "Timestep Shift",
                         tooltip="Shift the timestep distribution. Use the preview to see more details.")
        self.components.entry(frame, 5, 1, ui_state, "timestep_shift")

        # dynamic timestep shifting
        self.components.label(frame, 6, 0, "Dynamic Timestep Shifting",
                         tooltip="Dynamically shift the timestep distribution based on resolution. If enabled, the shifting parameters are taken from the model's scheduler configuration and Timestep Shift is ignored. Dynamic Timestep Shifting is not shown in the preview. Note: For Z-Image, the dynamic shifting parameters are likely wrong and unknown. Use with care or set your own, fixed shift.", wide_tooltip=True)
        self.components.switch(frame, 6, 1, ui_state, "dynamic_timestep_shifting")
