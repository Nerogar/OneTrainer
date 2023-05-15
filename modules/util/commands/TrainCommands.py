class TrainCommands:
    def __init__(self):
        self.__stop_command = False
        self.__sample_command = None

    def stop(self):
        self.__stop_command = True

    def get_stop_command(self) -> bool:
        return self.__stop_command

    def sample(self, sample_definition: dict):
        pass

    def get_and_reset_sample_command(self) -> dict:
        return self.__sample_command
