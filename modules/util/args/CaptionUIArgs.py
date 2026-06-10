import argparse
from typing import Any

from modules.util.args.BaseArgs import BaseArgs


class CaptionUIArgs(BaseArgs):
    dir: str
    include_subdirectories: bool

    def __init__(self, data: list[(str, Any, type, bool)]):
        super().__init__(data)

    @staticmethod
    def parse_args() -> 'CaptionUIArgs':
        parser = argparse.ArgumentParser(description="One Trainer Caption UI Script.")

        # @formatter:off

        parser.add_argument("--dir", type=str, required=False, default=None, dest="dir", help="The initial directory to load training data from")
        parser.add_argument("--include-subdirectories", action="store_true", required=False, default=False, dest="include_subdirectories", help="Whether to include subdirectories when processing samples")

        # @formatter:on

        args = CaptionUIArgs.default_values()
        args.from_dict(vars(parser.parse_args()))
        return args

    @staticmethod
    def default_values() -> 'CaptionUIArgs':
        data = []

        # name, default value, data type, nullable
        data.append(("dir", None, str, True))
        data.append(("include_subdirectories", False, bool, False))

        return CaptionUIArgs(data)
