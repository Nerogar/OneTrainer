from PIL import Image, ImageOps


def load_image(path: str, convert_mode: str = 'RGB') -> Image.Image:
    image = Image.open(path)
    image = ImageOps.exif_transpose(image)
    if convert_mode:
        image = image.convert(convert_mode)
    return image
