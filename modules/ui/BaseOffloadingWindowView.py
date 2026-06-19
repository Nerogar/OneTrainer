from modules.util.enum.GradientCheckpointingMethod import GradientCheckpointingMethod


class BaseOffloadingWindowView:
    def __init__(self, components):
        self.components = components

    def build_content(self, frame, controller, ui_state):
        self.components.label(frame, 0, 0, "Gradient checkpointing",
                              tooltip="Enables gradient checkpointing. This reduces memory usage, but increases training time")
        self.components.options(frame, 0, 1, [str(x) for x in list(GradientCheckpointingMethod)], ui_state,
                                "gradient_checkpointing")

        self.components.label(frame, 1, 0, "Async Offloading",
                              tooltip="Enables Asynchronous offloading.")
        self.components.switch(frame, 1, 1, ui_state, "enable_async_offloading")

        self.components.label(frame, 2, 0, "Offload Activations",
                              tooltip="Enables Activation Offloading")
        self.components.switch(frame, 2, 1, ui_state, "enable_activation_offloading")

        self.components.label(frame, 3, 0, "Layer offload fraction",
                              tooltip="Enables offloading of individual layers during training to reduce VRAM usage. Increases training time and uses more RAM. Only available if checkpointing is set to CPU_OFFLOADED. values between 0 and 1, 0=disabled")
        self.components.entry(frame, 3, 1, ui_state, "layer_offload_fraction")
