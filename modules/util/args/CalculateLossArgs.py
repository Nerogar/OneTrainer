import argparse
from typing import Any

from modules.util.args.BaseArgs import BaseArgs


class CalculateLossArgs(BaseArgs):
    config_path: str
    output_path: str

    def __init__(self, data: list[(str, Any, type, bool)]):
        super().__init__(data)

    @staticmethod
    def parse_args() -> 'CalculateLossArgs':
        parser = argparse.ArgumentParser(description="One Trainer Loss Calculation Script.")

        # @formatter:off

        parser.add_argument("--config-path", type=str, required=True, dest="config_path", help="The path to the config file")
        parser.add_argument("--output-path", type=str, required=True, dest="output_path", help="The path to the output file")

        # @formatter:on

        args = CalculateLossArgs.default_values()
        args.from_dict(vars(parser.parse_args()))
        return args

    @staticmethod
    def default_values() -> 'CalculateLossArgs':
        data = []

        # name, default value, data type, nullable
        data.append(("config_path", None, str, True))
        data.append(("output_path", "losses.json", str, False))

        return CalculateLossArgs(data)
