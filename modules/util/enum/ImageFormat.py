from modules.util.enum.BaseEnum import BaseEnum


class ImageFormat(BaseEnum):
    PNG = 'PNG'
    JPG = 'JPG'


    def pretty_print(self):
        return self.value

    def extension(self) -> str:
        match self:
            case ImageFormat.PNG:
                return ".png"
            case ImageFormat.JPG:
                return ".jpg"
            case _:
                return ""

    def pil_format(self) -> str:
        match self:
            case ImageFormat.PNG:
                return "PNG"
            case ImageFormat.JPG:
                return "JPEG"
            case _:
                return ""
