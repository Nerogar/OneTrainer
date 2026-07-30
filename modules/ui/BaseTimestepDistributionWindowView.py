




class BaseTimestepDistributionWindowView:
    def __init__(self, components):
        self.components = components

    def build_content(self, frame, controller, ui_state):
        # timestep distribution
        self.components.label(frame, 0, 0, "时间步分布",
                         tooltip="选择训练时的时间步采样函数",
                         wide_tooltip=True)
        self.components.options(frame, 0, 1, controller.get_distribution_options(), ui_state,
                           "timestep_distribution")

        # min noising strength
        self.components.label(frame, 1, 0, "最小噪声强度",
                         tooltip="训练最小噪声强度，有助于构图但会阻碍细节训练")
        self.components.entry(frame, 1, 1, ui_state, "min_noising_strength")

        # max noising strength
        self.components.label(frame, 2, 0, "最大噪声强度",
                         tooltip="训练最大噪声强度，可减少过拟合但降低样本对构图的影响")
        self.components.entry(frame, 2, 1, ui_state, "max_noising_strength")

        # noising weight
        self.components.label(frame, 3, 0, "噪声权重",
                         tooltip="控制时间步分布函数的权重参数")
        self.components.entry(frame, 3, 1, ui_state, "noising_weight")

        # noising bias
        self.components.label(frame, 4, 0, "噪声偏差",
                         tooltip="控制时间步分布函数的偏差参数")
        self.components.entry(frame, 4, 1, ui_state, "noising_bias")

        # timestep shift
        self.components.label(frame, 5, 0, "时间步偏移",
                         tooltip="偏移时间步分布，使用预览查看详情")
        self.components.entry(frame, 5, 1, ui_state, "timestep_shift")

        # dynamic timestep shifting
        self.components.label(frame, 6, 0, "动态时间步偏移",
                         tooltip="Dynamically shift the timestep distribution based on resolution. If enabled, the shifting parameters are taken from the model's scheduler configuration and Timestep Shift is ignored. Dynamic Timestep Shifting is not shown in the preview. For Ideogram, the shifting instead follows the model's own resolution-aware sampling schedule. Note: For Z-Image, the dynamic shifting parameters are likely wrong and unknown. Use with care or set your own, fixed shift.", wide_tooltip=True)
        self.components.switch(frame, 6, 1, ui_state, "dynamic_timestep_shifting")
