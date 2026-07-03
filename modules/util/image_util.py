from PIL import Image, ImageOps


def load_image(path: str, convert_mode: str = 'RGB') -> Image.Image:
    image = Image.open(path)
    image = ImageOps.exif_transpose(image)
    if convert_mode:
        image = image.convert(convert_mode)
    return image


def fit_image(image: Image.Image, max_w: int, max_h: int) -> Image.Image:
    """Scale ``image`` to fit within a ``(max_w, max_h)`` box, preserving aspect
    ratio, using LANCZOS resampling. Returns a new image."""
    scale = min(max_w / image.width, max_h / image.height)
    new_w = max(1, int(image.width * scale))
    new_h = max(1, int(image.height * scale))
    return image.resize((new_w, new_h), Image.Resampling.LANCZOS)


def load_fitted_image(path: str, max_w: int, max_h: int, convert_mode: str = 'RGB') -> Image.Image:
    """EXIF-transposed load (:func:`load_image`) fitted to a ``(max_w, max_h)``
    box — the one path UI thumbnails should use so they honor EXIF orientation
    like the rest of the app."""
    return fit_image(load_image(path, convert_mode), max_w, max_h)


def load_thumbnail(path: str, size: int, convert_mode: str = 'RGB') -> Image.Image:
    """Like :func:`load_fitted_image` for a square ``size`` box, but hints the
    decoder via ``Image.draft`` to read JPEGs at a reduced DCT scale first — much
    cheaper than decoding a multi-megapixel source only to shrink it. ``draft`` is
    a no-op for formats that don't support it, so correctness is unchanged."""
    image = Image.open(path)
    image.draft(convert_mode, (size, size))
    image = ImageOps.exif_transpose(image)
    if convert_mode:
        image = image.convert(convert_mode)
    return fit_image(image, size, size)
