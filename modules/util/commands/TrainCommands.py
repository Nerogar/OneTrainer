
from modules.util.config.SampleConfig import SampleConfig


class TrainCommands:
    def __init__(
            self,
            on_command=None# Callable[[TrainCommands], None] = lambda _: None
    ):
        self.reset()
        self.__stop_command = False
        self.__on_command = on_command

    def reset(self):
        #don't reset stop
        self.__sample_custom_commands = []
        self.__sample_default_command = False
        self.__backup_command = False
        self.__save_command = False

    def set_on_command(
            self,
            on_command#: Callable[[TrainCommands], None] = lambda _: None
    ):
        self.__on_command = on_command

    def get_and_reset_on_command(self):
        on_command = self.__on_command
        self.__on_command=None
        return on_command

    def stop(self):
        self.__stop_command = True
        if self.__on_command:
            self.__on_command(self)

    def get_stop_command(self) -> bool:
        return self.__stop_command

    def sample_custom(self, sample_params: SampleConfig):
        self.__sample_custom_commands.append(sample_params)
        if self.__on_command:
            self.__on_command(self)

    def get_and_reset_sample_custom_commands(self) -> list[SampleConfig]:
        sample_custom_commands = self.__sample_custom_commands
        self.__sample_custom_commands = []
        return sample_custom_commands

    def sample_default(self):
        self.__sample_default_command = True
        if self.__on_command:
            self.__on_command(self)

    def get_and_reset_sample_default_command(self) -> bool:
        sample_default_command = self.__sample_default_command
        self.__sample_default_command = False
        return sample_default_command

    def backup(self):
        self.__backup_command = True
        if self.__on_command:
            self.__on_command(self)

    def get_and_reset_backup_command(self) -> bool:
        backup_command = self.__backup_command
        self.__backup_command = False
        return backup_command

    def save(self):
        self.__save_command = True
        if self.__on_command:
            self.__on_command(self)

    def get_and_reset_save_command(self) -> bool:
        save_command = self.__save_command
        self.__save_command = False
        return save_command

    def merge(self, other):
        if other.get_stop_command():
            self.stop()
        for entry in other.get_and_reset_sample_custom_commands():
            self.sample_custom(entry)
        if other.get_and_reset_sample_default_command():
            self.sample_default()
        if other.get_and_reset_backup_command():
            self.backup()
        if other.get_and_reset_save_command():
            self.save()
