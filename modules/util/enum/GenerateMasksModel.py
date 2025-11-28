from modules.util.enum.BaseEnum import BaseEnum


class GenerateMasksModel(BaseEnum):
    CLIPSEG = 'CLIPSEG'
    REMBG = 'REMBG'
    REMBG_HUMAN = 'REMBG_HUMAN'
    COLOR = 'COLOR'

    def pretty_print(self):
        return {
            GenerateMasksModel.CLIPSEG: "CLIPSeg",
            GenerateMasksModel.REMBG: "RemBG",
            GenerateMasksModel.REMBG_HUMAN: "RemBG-Human",
            GenerateMasksModel.COLOR: "Hex Color"
        }[self]

class GenerateMasksAction(BaseEnum):
    REPLACE = 'REPLACE'
    FILL = 'FILL'
    ADD = 'ADD'
    SUBTRACT = 'SUBTRACT'
    BLEND = 'BLEND'

    def pretty_print(self):
        return {
            GenerateMasksAction.REPLACE: 'Replace all masks',
            GenerateMasksAction.FILL: 'Create if absent',
            GenerateMasksAction.ADD: 'Add to existing',
            GenerateMasksAction.SUBTRACT: 'Subtract from existing',
            GenerateMasksAction.BLEND: 'Blend with existing'
        }[self]
