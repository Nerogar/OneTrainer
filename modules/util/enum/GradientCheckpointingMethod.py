from modules.util.enum.BaseEnum import BaseEnum


class GradientCheckpointingMethod(BaseEnum):
    OFF = 'OFF'
    ON = 'ON'
    CPU_OFFLOADED = 'CPU_OFFLOADED'

    def pretty_print(self):
        return {
            GradientCheckpointingMethod.OFF: "Off",
            GradientCheckpointingMethod.ON: "On",
            GradientCheckpointingMethod.CPU_OFFLOADED: "CPU Offloaded",
        }[self]

    def enabled(self):
        return self == GradientCheckpointingMethod.ON \
            or self == GradientCheckpointingMethod.CPU_OFFLOADED

    def offload(self):
        return self == GradientCheckpointingMethod.CPU_OFFLOADED
