from enum import Enum


class AudioFormat(Enum):
    MP3 = 'MP3'

    def __str__(self):
        return self.value

    def extension(self) -> str:
        match self:
            case AudioFormat.MP3:
                return ".mp3"
            case _:
                return ""
