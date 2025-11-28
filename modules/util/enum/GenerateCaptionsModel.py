from modules.util.enum.BaseEnum import BaseEnum


class GenerateCaptionsModel(BaseEnum):
    BLIP = 'BLIP'
    BLIP2 = 'BLIP2'
    WD14_VIT_2 = 'WD14_VIT_2'

    def pretty_print(self):
        return {
            GenerateCaptionsModel.BLIP: "BLIP",
            GenerateCaptionsModel.BLIP2: "BLIP-2",
            GenerateCaptionsModel.WD14_VIT_2: "WD 1.4 ViT Tagger V2",
        }[self]


class GenerateCaptionsAction(BaseEnum):
    REPLACE = 'REPLACE'
    CREATE = 'CREATE'
    ADD = 'ADD'

    def pretty_print(self):
        return {
            GenerateCaptionsAction.REPLACE: 'Replace all captions',
            GenerateCaptionsAction.CREATE: 'Create if absent',
            GenerateCaptionsAction.ADD: 'Add as new line'
        }[self]
