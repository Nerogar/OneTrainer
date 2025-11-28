from modules.util.enum.BaseEnum import BaseEnum


class EMAMode(BaseEnum):
    OFF = 'OFF'
    GPU = 'GPU'
    CPU = 'CPU'

    def pretty_print(self):
        return {
            EMAMode.OFF: "Off",
            EMAMode.GPU: "GPU",
            EMAMode.CPU: "CPU",
        }[self]
