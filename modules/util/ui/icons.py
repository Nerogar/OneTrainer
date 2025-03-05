import os

import customtkinter as ctk
from PIL import Image, ImageEnhance

# Cache for loaded icons to avoid reloading the same icon
_icon_cache = {}


def load_icon(icon_name, size=(24, 24)):
    """
    Load a PNG icon for use in CTK buttons

    Parameters:
        icon_name: Name of the icon file without extension
        size: Tuple of (width, height) for the icon (default 24x24)

    Returns:
        CTkImage object or None if loading fails
    """
    cache_key = f"{icon_name}_{size[0]}x{size[1]}"

    if cache_key in _icon_cache:
        return _icon_cache[cache_key]

    # Find the project root resources directory
    icon_dir = os.path.join(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        ),
        "resources",
        "icons",
        "buttons"
    )

    # Simply look for a PNG file with the exact name
    png_path = os.path.join(icon_dir, f"{icon_name}.png")

    if os.path.exists(png_path):
        try:
            img = Image.open(png_path)

            resampling = Image.Resampling.BICUBIC if max(size) <= 24 else Image.Resampling.LANCZOS

            # Resize with the appropriate resampling method
            img = img.resize(size, resampling)

            # For very small icons, apply a slight sharpening to improve clarity
            if max(size) <= 24:
                enhancer = ImageEnhance.Sharpness(img)
                img = enhancer.enhance(1.10)  # Sharpen by 10%

            ctk_img = ctk.CTkImage(
                light_image=img, dark_image=img, size=size
            )
            _icon_cache[cache_key] = ctk_img
            return ctk_img
        except Exception as e:
            print(f"Error loading icon {icon_name}.png: {e}")

    print(f"Warning: No icon found for {icon_name}")

    # Create an empty placeholder image as fallback
    try:
        img = Image.new("RGBA", size, (0, 0, 0, 0))
        ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=size)
        _icon_cache[cache_key] = ctk_img
        return ctk_img
    except Exception as e:
        print(f"Error creating placeholder image: {e}")
        return None
