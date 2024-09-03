from enum import Enum


class GradientCheckpointingMethod(Enum):
    OFF = 'OFF'
    ON = 'ON'
    CPU_OFFLOADED = 'CPU_OFFLOADED'

    def __str__(self):
        return self.value

    def enabled(self):
        return self == GradientCheckpointingMethod.ON \
            or self == GradientCheckpointingMethod.CPU_OFFLOADED

    def offload(self):
        return self == GradientCheckpointingMethod.CPU_OFFLOADED
