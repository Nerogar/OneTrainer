import argparse


class GenerateCaptionsArgs:
    sample_dir: str
    initial_caption: str
    mode: str

    def __init__(self, args: dict):
        for (key, value) in args.items():
            setattr(self, key, value)

    @staticmethod
    def parse_args() -> 'GenerateCaptionsArgs':
        parser = argparse.ArgumentParser(description="One Trainer Generate Captions Script.")

        parser.add_argument("--sample-dir", type=str, required=True, dest="sample_dir", help="Directory where samples are located")
        parser.add_argument("--initial-caption", type=str, default='', required=False, dest="initial_caption", help="An initial caption to start generating from")
        parser.add_argument("--mode", type=str, default='fill', required=False, dest="mode", help="Either replace, fill, add or subtract")

        return GenerateCaptionsArgs(vars(parser.parse_args()))
