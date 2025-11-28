from modules.util.enum.BaseEnum import BaseEnum


class AudioFormat(BaseEnum):
    MP3 = 'MP3'

    def extension(self) -> str:
        match self:
            case AudioFormat.MP3:
                return ".mp3"
            case _:
                return ""
