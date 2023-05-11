class TrainCommands:
    def __init__(self):
        self.__stop_command = False
        self.__sample_command = None

    def stop(self):
        self.__stop_command = True

    def sample(self, sample_definition: dict):
        pass
