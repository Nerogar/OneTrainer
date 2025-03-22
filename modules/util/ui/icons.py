import logging
import threading
from pathlib import Path

import customtkinter as ctk
from PIL import Image, ImageEnhance, UnidentifiedImageError

logger = logging.getLogger(__name__)

_icon_cache = {}
_icon_lock = threading.Lock()  # Added because of an infamous user who opens 8 instances of OT.

def get_icon_path(icon_name: str) -> Path:
    current_file = Path(__file__)
    icon_dir = current_file.parent.parent.parent.parent / "resources" / "icons" / "buttons"
    return icon_dir / f"{icon_name}.png"

def load_icon(icon_name: str, size: tuple[int, int] = (24, 24)) -> ctk.CTkImage | None:
    """
    Load a PNG icon for use in CTK buttons.

    Parameters:
        icon_name (str): Name of the icon file without extension.
        size (tuple[int, int], optional): Desired size as (width, height). Defaults to (24, 24).

    Returns:
        ctk.CTkImage or None: A CTkImage object if the icon is successfully loaded or created, None otherwise.
    """
    cache_key = f"{icon_name}_{size[0]}x{size[1]}"

    with _icon_lock:
        if cache_key in _icon_cache:
            return _icon_cache[cache_key]

    png_path = get_icon_path(icon_name)

    if png_path.exists():
        try:
            with Image.open(png_path) as file_img:
                img = file_img.convert("RGBA")
                resampling = Image.Resampling.BICUBIC if max(size) <= 24 else Image.Resampling.LANCZOS
                img = img.resize(size, resampling)

                if max(size) <= 24:
                    enhancer = ImageEnhance.Sharpness(img)
                    img = enhancer.enhance(1.25)
                # Copy the image so it remains available after the context closes
                final_img = img.copy()

            ctk_img = ctk.CTkImage(light_image=final_img, dark_image=final_img, size=size)
            with _icon_lock:
                _icon_cache[cache_key] = ctk_img
            return ctk_img

        except (OSError, UnidentifiedImageError) as e:
            logger.error(f"Error loading icon {icon_name}.png: {e}")
        except ValueError as e:
            logger.error(f"Error processing icon {icon_name}.png: {e}")

    logger.warning(f"Warning: No icon found for {icon_name}")

    try:
        img = Image.new("RGBA", size, (0, 0, 0, 0))
        ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=size)
        with _icon_lock:
            _icon_cache[cache_key] = ctk_img
        return ctk_img

    except Exception as e:
        logger.error(f"Error creating placeholder image: {e}")
        return None
