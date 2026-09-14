import os

from modules.util import path_util

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


class SelectMaskPath(
    PipelineModule,
    RandomAccessPipelineModule,
):
    """Derives the mask path of an image, cycling through numbered mask variants.

    An image can have several masks: the original '<name>-masklabel.png' plus
    '<name>-masklabel1.png' up to '<name>-masklabel9.png'. Variation N of the image
    cache uses variant N, wrapping around, so the mask varies alongside the
    brightness and flip augmentations that are already keyed on the variation index.

    With a single mask, or none at all, this behaves like ModifyPath.
    """

    def __init__(self, in_name: str, out_name: str):
        super().__init__()

        self.in_name = in_name
        self.out_name = out_name

        self.mask_paths = []

    def length(self) -> int:
        return self._get_previous_length(self.in_name)

    def get_inputs(self) -> list[str]:
        return [self.in_name]

    def get_outputs(self) -> list[str]:
        return [self.out_name]

    def start(self, variation: int):
        length = self._get_previous_length(self.in_name)
        if len(self.mask_paths) == length:
            # image paths don't change between epochs, only scan once
            return

        self.mask_paths = []

        # one listing per directory instead of a stat per candidate, which would be
        # ten extra syscalls for every sample in the dataset
        listed_directories = {}

        for index in range(length):
            image_path = self._get_previous_item(variation, self.in_name, index)

            directory = os.path.dirname(image_path)
            if directory not in listed_directories:
                try:
                    listed_directories[directory] = set(os.listdir(directory))
                except OSError:
                    listed_directories[directory] = set()
            filenames = listed_directories[directory]

            image_name = os.path.splitext(os.path.basename(image_path))[0]
            variants = [
                path_util.mask_path_for(image_path, variant)
                for variant in range(path_util.MAX_MASK_VARIANTS + 1)
                if image_name + path_util.mask_postfix(variant) + path_util.MASK_EXTENSION in filenames
            ]

            # without any mask, keep the canonical path so LoadImage returns None and
            # the pipeline falls back to the generated fully included mask
            self.mask_paths.append(variants if variants else [path_util.mask_path_for(image_path)])

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        variants = self.mask_paths[index]

        return {
            self.out_name: variants[variation % len(variants)],
        }
