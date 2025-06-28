import numpy as np
from pepedpid import dpid_resize as pepedpid_resize_impl
from PIL import Image, ImageOps


# generic image loading helper
def load_image(path: str, convert_mode: str = 'RGB') -> Image.Image:
    image = Image.open(path)
    image = ImageOps.exif_transpose(image)
    if convert_mode:
        image = image.convert(convert_mode)
    return image

def dpid_resize(
    pil_image: Image.Image, size: tuple[int, int], lambda_val: float = 0.51
) -> Image.Image:
    """
    Resizes a PIL image using the pepedpid library.

    Args:
        pil_image: The source image as a PIL Image object.
        size: A tuple containing the (width, height) for the output image.
        lambda_val: Controls detail enhancement.
                    λ ≈ 0.0 — maximum smoothing, the image will be soft.
                    λ ≈ 1.0 — maximum detail preservation, resulting in a sharp image.
                    Recommended value: 0.5 — balance between smoothness and sharpness.

    Returns:
        The resized image as a PIL Image object.
    """
    # Convert PIL image to numpy array, normalize to [0, 1] float32
    input_array = np.array(pil_image, dtype=np.float32) / 255.0

    # pepedpid expects h, w as separate arguments
    out_width, out_height = size

    # Apply DPID resizing
    resized_array = pepedpid_resize_impl(input_array, h=out_height, w=out_width, l=lambda_val)

    # Convert back to uint8 [0, 255] and then to PIL Image
    resized_array_uint8 = np.clip(resized_array * 255.0, 0, 255).astype(np.uint8)

    return Image.fromarray(resized_array_uint8)
