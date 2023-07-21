import argparse


class CreateTrainFilesArgs:
    concepts_output_destination: str
    samples_output_destination: str

    def __init__(self, args: dict):
        for (key, value) in args.items():
            setattr(self, key, value)

    @staticmethod
    def parse_args() -> 'CreateTrainFilesArgs':
        parser = argparse.ArgumentParser(description="One Trainer Sampling Script.")

        parser.add_argument("--concepts-output-destination", type=str, required=False, default=None, dest="concepts_output_destination", help="The destination filename to save a default concepts file")
        parser.add_argument("--samples-output-destination", type=str, required=False, default=None, dest="samples_output_destination", help="The destination filename to save a default samples file")

        return CreateTrainFilesArgs(vars(parser.parse_args()))
