import argparse


class CaptionUIArgs:
    dir: str

    def __init__(self, args: dict):
        for (key, value) in args.items():
            setattr(self, key, value)

    @staticmethod
    def parse_args() -> 'CaptionUIArgs':
        parser = argparse.ArgumentParser(description="One Trainer Caption UI Script.")

        parser.add_argument("--dir", type=str, required=False, default=None, dest="dir", help="The initial directory to load training data from")

        return CaptionUIArgs(vars(parser.parse_args()))
