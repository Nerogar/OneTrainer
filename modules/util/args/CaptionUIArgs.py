import argparse
from typing import Any

from modules.util.args.BaseArgs import BaseArgs


class CaptionUIArgs(BaseArgs):
    dir: str

    def __init__(self, data: list[(str, Any, type, bool)]):
        super(CaptionUIArgs, self).__init__(data)

    @staticmethod
    def parse_args() -> 'CaptionUIArgs':
        parser = argparse.ArgumentParser(description="One Trainer Caption UI Script.")

        # @formatter:off

        parser.add_argument("--dir", type=str, required=False, default=None, dest="dir", help="The initial directory to load training data from")

        # @formatter:on

        args = CaptionUIArgs.default_values()
        args.from_dict(vars(parser.parse_args()))
        return args

    @staticmethod
    def default_values() -> 'CaptionUIArgs':
        data = []

        # name, default value, data type, nullable
        data.append(("dir", None, str, True))

        return CaptionUIArgs(data)
