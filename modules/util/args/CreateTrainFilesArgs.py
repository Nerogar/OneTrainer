import argparse
from typing import Any

from modules.util.args.BaseArgs import BaseArgs


class CreateTrainFilesArgs(BaseArgs):
    config_output_destination: str
    concepts_output_destination: str
    samples_output_destination: str

    def __init__(self, data: list[(str, Any, type, bool)]):
        super().__init__(data)

    @staticmethod
    def parse_args() -> 'CreateTrainFilesArgs':
        parser = argparse.ArgumentParser(description="One Trainer Create Train Files Script.")

        # @formatter:off

        parser.add_argument("--config-output-destination", type=str, required=False, default=None, dest="config_output_destination", help="The destination filename to save a default config file")
        parser.add_argument("--concepts-output-destination", type=str, required=False, default=None, dest="concepts_output_destination", help="The destination filename to save a default concepts file")
        parser.add_argument("--samples-output-destination", type=str, required=False, default=None, dest="samples_output_destination", help="The destination filename to save a default samples file")

        # @formatter:on

        args = CreateTrainFilesArgs.default_values()
        args.from_dict(vars(parser.parse_args()))
        return args


    @staticmethod
    def default_values():
        data = []

        data.append(("config_output_destination", None, str, True))
        data.append(("concepts_output_destination", None, str, True))
        data.append(("samples_output_destination", None, str, True))

        return CreateTrainFilesArgs(data)
