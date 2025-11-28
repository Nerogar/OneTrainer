import functools
import os
import random
import threading
import traceback
import uuid
from multiprocessing import Pool
from pathlib import Path

from modules.ui.models.SingletonConfigModel import SingletonConfigModel
from modules.util import image_util, path_util
from modules.util.enum.ImageMegapixels import ImageMegapixels
from modules.util.enum.ImageOperations import ImageOperations
from modules.util.enum.ImageOptimization import ImageOptimization

import imagesize
import oxipng
from PIL import Image, ImageColor

# multiprocessing.Pool requires pickable functions, which must be defined outside classes.
# https://pypi.org/project/pathos/ uses dill as serialization engine, allowing more flexibility, in case we want to switch to something more mainteinable.

def _verify_image(file):
    """
            Verify a single image file for corruption. Raises ValueError on failure.

            Note: The image is opened twice because Pillow's .verify() invalidates the image object.
            """
    file_path = Path(file)
    valid = False
    try:
        with Image.open(file_path) as img:
            img.verify()
        with Image.open(file_path) as img:
            img.load()
            if hasattr(img, "getpixel"):
                img.getpixel((0, 0))
        valid = True
    except Exception:
        valid = False

    return valid

def _process_alpha_image(bg_color_tuple, file):
    with Image.open(file) as img:
        if img.mode not in ("RGBA", "LA"):
            return False

        background = Image.new("RGB", img.size, bg_color_tuple)
        background.paste(img, (0, 0), img)
        background.save(str(file))
        return True

def _resize_large_image(target_pixels, file):
    width, height = imagesize.get(file)
    if width * height <= target_pixels:
        return False

    new_width, new_height = ImageModel.calculate_dimensions_for_megapixels(
        width, height, target_pixels
    )
    reduction_factor = width / new_width

    with Image.open(file) as img:
        resample_filter = (
            Image.Resampling.LANCZOS
            if hasattr(Image, "Resampling")
            else Image.LANCZOS
        )

        if reduction_factor >= 3 and hasattr(image_util, "dpid_resize"):
            resized_img = image_util.dpid_resize(img, (new_width, new_height))
        else:
            resized_img = img.resize((new_width, new_height), resample=resample_filter)

        save_kwargs = {}
        if file.suffix.lower() in [".jpg", ".jpeg"]:
            save_kwargs["quality"] = 95
            if "icc_profile" in img.info:
                save_kwargs["icc_profile"] = img.info["icc_profile"]
            if "exif" in img.info:
                save_kwargs["exif"] = img.info["exif"]
        elif file.suffix.lower() == ".png":
            save_kwargs["compress_level"] = 4

        if resized_img.mode == "P":
            resized_img = resized_img.convert("RGB")

        resized_img.save(str(file), **save_kwargs)
    return True


def _optimize_png(file):
    original_size = file.stat().st_size
    oxipng.optimize(file, level=5, fix_errors=True)
    new_size = file.stat().st_size
    bytes_saved = original_size - new_size
    if bytes_saved > 0:
        return True, bytes_saved
    else:
        return False, 0

def _is_lossless_check(file, img, check_lossless):
    if not check_lossless:
        return False
    else:
        return file.suffix.lower() in {".jpg", ".jpeg"}

def _convert_image(format_options, file):
    # Use local variables for performance and clarity
    format_ext = format_options["format_ext"]
    pil_format = format_options["pil_format"]
    lossless_extensions = format_options.get("lossless_extensions", set())
    quality = format_options.get("quality", 90)
    save_kwargs_base = format_options.get("save_kwargs", {})

    original_size = file.stat().st_size
    new_path = file.with_suffix(format_ext)

    with Image.open(file) as img:
        is_lossless = file.suffix.lower() in lossless_extensions or _is_lossless_check(file, img, format_options["check_lossless"])

        save_kwargs = save_kwargs_base.copy()
        save_kwargs["quality"] = quality
        if is_lossless:
            save_kwargs["lossless"] = True

        img.save(new_path, pil_format, **save_kwargs)

    if not new_path.exists():
        return "errors", (file.name, "Failed to save new file.")

    new_size = new_path.stat().st_size
    bytes_saved = original_size - new_size

    if new_size < original_size:
        file.unlink()
        return True, bytes_saved
    else:
        new_path.unlink()
        return False, 0


class ImageModel(SingletonConfigModel):
    def __init__(self):
        super().__init__({
            "directory": "",
            "verify_images": False,
            "sequential_rename": False,
            "process_alpha": False,
            "resize_large_images": False,
            "optimization_type": ImageOptimization.NONE,
            "resize_megapixels": ImageMegapixels.COMPUTE_PROOF_MEGAPIXEL_THRESHOLD,
            "resize_custom_megapixels": 4,
            "alpha_bg_color": "#ffffff",
        })

        self.pool = None
        self.abort_flag = threading.Event()
        self.progress_fn = None

    def process_files(self, progress_fn=None):
        with self.critical_region_read():
            directory = self.get_state("directory")
            self.progress_fn = progress_fn

            if self.pool is None:
                self.pool = Pool()

            if os.path.isdir(directory):
                path = Path(directory)
                files = [f for f in path.iterdir() if f.is_file()]
                self.log("info", f"Found {len(files)} files in {directory}")

                self.__run_operations(files)

    def terminate_pool(self):
        if self.pool is not None:
            self.pool.terminate()
            self.pool.join()
            self.pool = None

    def __run_operations(self, files):
        operations = []
        if self.get_state("verify_images"):
            operations.append(ImageOperations.VERIFY_IMG)
        if self.get_state("sequential_rename"):
            operations.append(ImageOperations.SEQUENTIAL_RENAME)
        if self.get_state("process_alpha"):
            operations.append(ImageOperations.PROCESS_ALPHA)
        if self.get_state("resize_large_images"):
            operations.append(ImageOperations.RESIZE_LARGE_IMG)

        opt = self.get_state("optimization_type")
        if opt == ImageOptimization.PNG:
            operations.append(ImageOperations.OPTIMIZE_PNG)
        elif opt == ImageOptimization.WEBP:
            operations.append(ImageOperations.CONVERT_WEBP)
        elif opt == ImageOptimization.JXL:
            operations.append(ImageOperations.CONVERT_JXL)

        # The correctness of these operations relies on the insertion order, as SEQUENTIAL_RENAME, CONVERT_WEBP and CONVERT_JXL change the list of files.
        # In the original implementation (and this one as well) conversion is performed last, so we need to keep track of changed files only for SEQUENTIAL_RENAME.

        total_ops = len(operations)
        i = 0
        while not self.abort_flag.is_set() and len(operations) > 0:
            op= operations.pop(0)
            i += 1
            if self.progress_fn is not None:
                self.progress_fn({"status": op.pretty_print(), "value": i, "max_value": total_ops, "data": f"{op.pretty_print()}..."})
            try:
                if op == ImageOperations.VERIFY_IMG:
                    self.__verify_images(files)
                elif op == ImageOperations.SEQUENTIAL_RENAME:
                    files = self.__rename_files_sequentially(files)
                elif op == ImageOperations.PROCESS_ALPHA:
                    self.__process_alpha_images(files)
                elif op == ImageOperations.RESIZE_LARGE_IMG:
                    self.__resize_large_images(files)
                elif op == ImageOperations.OPTIMIZE_PNG:
                    self.__optimize_pngs(files)
                elif op == ImageOperations.CONVERT_WEBP:
                    self.__convert_to_webp(files)
                elif op == ImageOperations.CONVERT_JXL:
                    self.__convert_to_jpegxl(files)
            except Exception:
                if self.progress_fn is not None:
                    self.progress_fn({"status": f"Error during {op.pretty_print().lower()}"})
                    self.log("critical", traceback.format_exc())

        if self.abort_flag.is_set():
            if self.progress_fn is not None:
                self.progress_fn({"status": "Processing aborted"})
        else:
            if self.progress_fn is not None:
                self.progress_fn({"status": "Processing complete", "value": 0, "max_value": 0})

    @staticmethod
    def calculate_dimensions_for_megapixels(
            original_width: int, original_height: int, target_pixels: int
    ) -> tuple[int, int]:
        """Calculates new dimensions to fit an image within a pixel budget."""
        original_pixels = original_width * original_height
        if original_pixels <= target_pixels:
            return original_width, original_height

        scale_factor = (target_pixels / original_pixels) ** 0.5
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)
        return new_width, new_height

    @staticmethod
    def _filter_is_image(f: Path) -> bool:
        """Filter for supported image files."""
        return path_util.is_supported_image_extension(f.suffix)

    @staticmethod
    def _filter_images_and_skip_masks(f: Path) -> bool:
        """Filter for supported image files, excluding mask files."""
        return path_util.is_supported_image_extension(f.suffix) and not f.stem.endswith("-masklabel")

    @staticmethod
    def _file_filter(f, skip_extensions):
        return (
                path_util.is_supported_image_extension(f.suffix)
                and f.suffix.lower() not in skip_extensions
                and "-" not in f.stem  # Skip mask files
        )

    def __verify_images(self, files):
        # We want to check only images.
        files = [f for f in files if self._filter_is_image(f)]

        # TODO: maybe this should also verify that there are no ambiguous images (e.g., "1.png" and "1.jpeg" in the same folder)

        is_valid = self.pool.map(_verify_image, files)

        total = len(is_valid)
        valid = len([v for v in is_valid if v])
        invalid = total - valid

        # If there is no progress callback, there is no reason to compute the output. Note that _verify_image may still be reimplemented in the future to attempt a file fix, therefore we should still map files to it.
        if self.progress_fn is not None:
            for file, valid in zip(files, is_valid, strict=True):
                if valid:
                    self.progress_fn({"data": f"✓ {file.name} is valid"})
                else:
                    self.progress_fn({"data": f"✗ {file.name} is CORRUPTED"}) # TODO: the original implementation also passed the exception message.

            self.progress_fn({"data": f"Checked {total} files: {valid} valid, {invalid} invalid"})

    def __rename_files_sequentially(self, files):
        outfiles = files
        if len(files) > 0:
            # TODO: This does not always work (after processing multiple files are deleted, and caption/mask become associated with the wrong image), but could NOT reproduce bug with:
            # dataset with images + masks + captions
            # dataset with some captions missing
            # dataset with some masks missing
            # dataset with mixed png, jpeg # TODO: do we also want to add an operation to convert all images to the same format (other than webp/jxl)?
            # Probably it was an edge case like "img.png + img.jpeg + img.txt + img-masklabel.png", somehow desynchronizing all the other valid triplets?
            # The other possible cause may be an incorrect exception handling.

            # TODO Improvement: should unlink() be replaced by the OS' send to recycling bin? https://pypi.org/project/Send2Trash/


            groups = {}

            for f in files:
                stem = f.stem

                key = stem.removesuffix("-masklabel") if stem.endswith("-masklabel") else stem

                if key not in groups:
                    groups[key] = {"image": None, "caption": None, "masks": []}

                if stem.endswith("-masklabel"):
                    groups[key]["masks"].append(f)  # TODO: Why can it be the case that we have multiple masks?
                elif self._filter_is_image(f):
                    groups[key]["image"] = f
                elif f.suffix.lower() == ".txt":
                    groups[key]["caption"] = f


            image_groups = {stem: data for stem, data in groups.items() if data["image"] is not None}

            if self.progress_fn is not None:
                self.progress_fn({"data": f"Found {len(image_groups)} images and their associated files to rename."})

            # Process files only if there is at least one image, and they are not already sorted.
            if len(image_groups) > 0:
                if not any(
                (not f.isdigit()) or (int(f) != i + 1)
                for i, f in enumerate(sorted(image_groups.keys()))
                ):
                    if self.progress_fn is not None:
                        self.progress_fn({"data": "Files are already named sequentially. No action needed."})
                else:
                    # TODO: from what I understand, the original intended behavior was to undo EVERY renaming, in case of an OSError.
                    # I think a best-effort strategy is more reasonable: if a single sample fails, rollback only its files (image, caption and masks), and stop there.
                    # This is because if you get an error during renaming, either there are permission issues (on some files), or the directory is no longer writeable/mounted (and in that case rollbacking would fail, and we would be in an inconsistent state anyway).
                    # The only goal of the rollback mechanism is to attempt to guarantee that no (image, caption, mask) triplet association is lost due to partial renamings.
                    outfiles = []

                    renaming = []
                    for i, img in enumerate(sorted(image_groups), start=1):
                        tmp = image_groups[img]
                        tmp_name = str(uuid.uuid4().hex)
                        renaming.extend([(tmp["image"], tmp["image"].with_name(f"{tmp_name}{tmp['image'].suffix}"), tmp["image"].with_name(f"{i}{tmp['image'].suffix}"))])
                        if tmp["caption"] is not None:
                            renaming.extend([(tmp["caption"], tmp["caption"].with_name(f"{tmp_name}{tmp['caption'].suffix}"), tmp["caption"].with_name(f"{i}{tmp['caption'].suffix}"))])
                        for mask in tmp["masks"]:
                            renaming.extend([(mask, mask.with_name(f"{tmp_name}-masklabel{mask.suffix}"), mask.with_name(f"{i}-masklabel{mask.suffix}"))])

                    errored = False
                    # Rename all the samples (image, caption and mask) in order. To avoid overwriting samples which already have integer names, we need two passes.
                    # First pass: source to temp name.
                    last_before_error = 0
                    try:
                        for j, (src, tmp_dest, _) in enumerate(renaming):
                            self.progress_fn({"data": f"Renaming {src.name} to {tmp_dest.name}..."})
                            src.rename(tmp_dest)
                            last_before_error = j

                    except OSError:
                        errored = True
                        try:
                            for src2, dest2, _ in renaming[:-last_before_error]:
                                dest2.rename(src2)
                        except OSError:
                            if self.progress_fn is not None:
                                self.progress_fn({"status": "Critical failure during rename",
                                                  "data": f"OSError while attempting rollback. Is {dest2.path} still accessible?"})
                                self.log("critical", traceback.format_exc())

                        if self.progress_fn is not None:
                            self.progress_fn({"status": "Rename failed, successfully rolled back.", "data": f"Rename failed for file {src.name}"})

                    if not errored:
                        # Second pass: temp name to destination.
                        last_before_error = 0
                        try:
                            for j, (_, tmp_dest, final_dest) in enumerate(renaming):
                                self.progress_fn({"data": f"Renaming {tmp_dest.name} to {final_dest.name}..."})
                                tmp_dest.rename(final_dest)
                                outfiles.append(final_dest)
                                last_before_error = j

                        except OSError:
                            try:
                                for _, src2, dest2 in renaming[:-last_before_error]:
                                    dest2.rename(src2)
                                outfiles = outfiles[:-last_before_error]
                            except OSError:
                                if self.progress_fn is not None:
                                    self.progress_fn({"status": "Critical failure during rename",
                                                      "data": f"OSError while attempting rollback. Is {dest2.path} still accessible?"})
                                    self.log("critical", traceback.format_exc())

                            if self.progress_fn is not None:
                                self.progress_fn({"status": "Rename failed, successfully rolled back.",
                                                  "data": f"Rename failed for file {tmp_dest.name}"})

        return outfiles

    def __process_alpha_images(self, files):
        bg_color_str = self.get_state("alpha_bg_color")

        files = [f for f in files if self._filter_images_and_skip_masks(f)]

        if self.progress_fn is not None:
            self.progress_fn({"data": "Processing transparent images (excluding mask files)..."})


        try:
            if bg_color_str.lower() in ("-1", "random"):
                r, g, b = (random.randint(0, 255) for _ in range(3))
                bg_color_tuple = (r, g, b)

                if self.progress_fn is not None:
                    self.progress_fn({"data": f"Using random background color: #{r:02x}{g:02x}{b:02x}"})
            else:
                color = ImageColor.getrgb(bg_color_str)
                # Ensure we have a 3-channel RGB tuple for the background
                bg_color_tuple = color[:3]

            if self.progress_fn is not None:
                self.progress_fn({"data": f"Using background color: {bg_color_str} (RGB: {bg_color_tuple})"})
        except (ValueError, TypeError) as e:
            if self.progress_fn is not None:
                self.progress_fn({"data": f"Invalid color '{bg_color_str}': {e}. Using white instead"})

            bg_color_tuple = (255, 255, 255)  # Fallback to White in case of unknown color.

        result = self.pool.map(functools.partial(_process_alpha_image, bg_color_tuple), files)
        total = len(result)
        processed = len([r for r in result if r])
        skipped = total - processed

        if self.progress_fn is not None:
            self.progress_fn({"data": f"Transparency processing complete: {total} total, {processed} processed, {skipped} skipped"})

    def __resize_large_images(self, files):
        files = [f for f in files if self._filter_is_image(f)] # TODO: original implementation skipped masks, but it does not make sense to rescale images, but not masks.

        mp = self.get_state("resize_megapixels")
        if mp == ImageMegapixels.CUSTOM:
            mp = ImageMegapixels.ONE_MEGAPIXEL.value * self.get_state("resize_custom_megapixels")
        else:
            mp = mp.value

        if self.progress_fn is not None:
            self.progress_fn({"data": f"Starting resizing of large images... Target: {mp / ImageMegapixels.ONE_MEGAPIXEL.value:.1f}MP"})

        result = self.pool.map(functools.partial(_resize_large_image, mp), files)
        total = len(result)
        processed = len([r for r in result if r])
        skipped = total - processed

        if self.progress_fn is not None:
            self.progress_fn({"data": f"Image resizing complete: {total} total, {processed} processed, {skipped} skipped"})


    def __optimize_pngs(self, files):
        files = [f for f in files if f.suffix == ".png"]
        result = self.pool.map(_optimize_png, files)
        total = len(result)
        processed = len([r for r in result if r[0]])
        skipped = total - processed

        bytes_saved = sum([r[1] for r in result])
        avg_bytes_saved = bytes_saved / total if total > 0 else 0

        if self.progress_fn is not None:
            self.progress_fn({"data": f"Completed optimization: {processed} PNGs optimized, {skipped} skipped. Saved {bytes_saved} bytes ({avg_bytes_saved} average bytes per file)"})

    def __convert_to_webp(self, files):
        """Convert images to WebP format using the generic converter."""
        format_options = {
            "format_ext": ".webp",
            "pil_format": "WEBP",
            "lossless_extensions": {".png", ".tiff", ".tif", ".bmp"},
            "check_lossless": False,
            "quality": 90,
        }
        skip_extensions = {".webp", ".jxl", ".avif"}
        self.__convert_image_format(
            files,
            "WebP",
            skip_extensions,
            format_options,
        )

    def __convert_to_jpegxl(self, files):
        """Convert images to JPEG XL format using the generic converter."""
        format_options = {
            "format_ext": ".jxl",
            "pil_format": "JXL",
            "lossless_extensions": set(),
            "check_lossless": True,
            "quality": 90,
        }
        skip_extensions = {".jxl"}
        self.__convert_image_format(
            files,
            "JPEG XL",
            skip_extensions,
            format_options
        )

    def __convert_image_format(
        self,
        files: list[Path],
        target_format: str,
        skip_extensions: set,
        format_options: dict
    ) -> None:
        """Generic image conversion function for multiple formats."""

        if self.progress_fn is not None:
            self.progress_fn({"data": f"Starting conversion to {target_format} format..."})

        files = [f for f in files if self._file_filter(f, skip_extensions)]

        result = self.pool.map(functools.partial(_convert_image, format_options), files)
        total = len(result)
        processed = len([r for r in result if r[0]])
        skipped = total - processed

        bytes_saved = sum([r[1] for r in result])
        avg_bytes_saved = bytes_saved / total

        if self.progress_fn is not None:
            self.progress_fn({"data": f"Completed optimization: {processed} PNGs optimized, {skipped} skipped. Saved {bytes_saved} bytes ({avg_bytes_saved} average bytes per file)"})
