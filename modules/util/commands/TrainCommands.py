from modules.util.config.SampleConfig import SampleConfig


class TrainCommands:
    def __init__(self):
        self.__stop_command = False
        self.__sample_custom_commands = []
        self.__sample_default_command = False
        self.__backup_command = False
        self.__save_command = False

    def stop(self):
        self.__stop_command = True

    def get_stop_command(self) -> bool:
        return self.__stop_command

    def sample_custom(self, sample_params: SampleConfig):
        self.__sample_custom_commands.append(sample_params)

    def get_and_reset_sample_custom_commands(self) -> list[SampleConfig]:
        sample_custom_commands = self.__sample_custom_commands
        self.__sample_custom_commands = []
        return sample_custom_commands

    def sample_default(self):
        self.__sample_default_command = True

    def get_and_reset_sample_default_command(self) -> bool:
        sample_default_command = self.__sample_default_command
        self.__sample_default_command = False
        return sample_default_command

    def backup(self):
        self.__backup_command = True

    def get_and_reset_backup_command(self) -> bool:
        backup_command = self.__backup_command
        self.__backup_command = False
        return backup_command

    def save(self):
        self.__save_command = True

    def get_and_reset_save_command(self) -> bool:
        save_command = self.__save_command
        self.__save_command = False
        return save_command
