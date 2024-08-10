import argparse
from typing import Any

from modules.util.args.BaseArgs import BaseArgs


class TrainArgs(BaseArgs):
    config_path: str

    def __init__(self, data: list[(str, Any, type, bool)]):
        super(TrainArgs, self).__init__(data)

    @staticmethod
    def parse_args() -> 'TrainArgs':
        parser = argparse.ArgumentParser(description="One Trainer Training Script.")

        # @formatter:off

        parser.add_argument("--config-path", type=str, required=True, dest="config_path", help="The path to the config file")

        # @formatter:on

        args = TrainArgs.default_values()
        args.from_dict(vars(parser.parse_args()))
        return args

    @staticmethod
    def default_values() -> 'TrainArgs':
        data = []

        # name, default value, data type, nullable
        data.append(("config_path", None, str, True))

        return TrainArgs(data)
