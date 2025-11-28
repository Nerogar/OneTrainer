from modules.util.enum.BaseEnum import BaseEnum


class PromptSource(BaseEnum):
    SAMPLE = 'sample'
    CONCEPT = 'concept'
    FILENAME = 'filename'

    def pretty_print(self):
        return {
            PromptSource.SAMPLE: "From text file per sample",
            PromptSource.CONCEPT: "From single text file",
            PromptSource.FILENAME: "From image file name"
        }[self]
