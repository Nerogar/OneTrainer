from modules.util.params.SampleParams import SampleParams


class TrainCommands:
    def __init__(self):
        self.__stop_command = False
        self.__sample_command = None

    def stop(self):
        self.__stop_command = True

    def get_stop_command(self) -> bool:
        return self.__stop_command

    def sample(self, sample_params: SampleParams):
        pass

    def get_and_reset_sample_command(self) -> SampleParams:
        return self.__sample_command
