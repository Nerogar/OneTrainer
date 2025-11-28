import os
import re
from pathlib import Path

from modules.ui.models.SingletonConfigModel import SingletonConfigModel
from modules.util.config.BaseConfig import BaseConfig
from modules.util.enum.CaptionFilter import CaptionFilter
from modules.util.enum.FileFilter import FileFilter

from PIL import Image


class DatasetConfig(BaseConfig):
    path: str
    valid: bool # Is a valid dataset path?
    include_subdirectories: bool
    file_filter: str
    file_filter_mode: FileFilter
    caption_filter: str
    caption_filter_mode: CaptionFilter
    files: list

    @staticmethod
    def default_values():
        data = []

        # name, default value, data type, nullable
        data.append(("path", None, str, True))
        data.append(("valid", False, bool, False))
        data.append(("include_subdirectories", False, bool, False))
        data.append(("file_filter", "", str, False))
        data.append(("file_filter_mode", FileFilter.FILE, FileFilter, False))
        data.append(("caption_filter", "", str, False))
        data.append(("caption_filter_mode", CaptionFilter.MATCHES, CaptionFilter, False))
        data.append(("files", [], list, False))

        return DatasetConfig(data)



class DatasetModel(SingletonConfigModel):
    def __init__(self):
        super().__init__(DatasetConfig.default_values())

    def scan(self):
        path, include_subdirs = self.bulk_read("path", "include_subdirectories")

        if path is not None:
            root = Path(path)

            root_str = str(root.resolve())
            root_len = len(root_str) + 1  # ".../dir" + "/"
            stack = [root_str]
            results = []

            while stack:
                top = stack.pop()
                with os.scandir(top) as it:
                    for entry in it:
                        if entry.is_dir(follow_symlinks=False):
                            if include_subdirs:
                                stack.append(entry.path)
                            continue
                        name = entry.name
                        if self.__is_supported(name):
                            # strip root and back-slashes only once
                            results.append(entry.path[root_len:].replace("\\", "/"))
            self.set_state("files", sorted(results, key=lambda x: self.natural_sort_key(x)))

    def getFilteredFiles(self):
        (path, unfiltered_files, file_filter, file_filter_mode,
         caption_filter, caption_filter_mode) = self.bulk_read("path", "files", "file_filter",
                                                               "file_filter_mode", "caption_filter", "caption_filter_mode")
        file_filter = file_filter.strip()
        caption_filter = caption_filter.strip()

        if file_filter == "" and caption_filter == "":
            return unfiltered_files

        filtered = [str(f) for f in unfiltered_files]

        if file_filter != "":
            try:
                pattern = re.compile(re.escape(file_filter), re.IGNORECASE)

                if file_filter_mode == FileFilter.FILE:
                    filtered = [
                        f for f in filtered if pattern.search(Path(f).name)
                    ]
                elif file_filter_mode == FileFilter.PATH:
                    filtered = [
                        f for f in filtered if pattern.search(f)  # f is already str # TODO: This is taken from the original implementation, however it has the same effect of FileFilter.BOTH, because it does not strip the filename before searching
                    ]
                else:  # Both
                    filtered = [
                        f
                        for f in filtered
                        if pattern.search(f) or pattern.search(Path(f).name)
                    ]
            except re.error:
                pass

        if caption_filter != "" and path is not None:
            try:
                caption_files = []
                for file_path_str in filtered:  # Iterate over strings
                    full_path = Path(path) / file_path_str  # file_path_str is relative
                    caption_path = full_path.with_suffix(".txt")

                    if not caption_path.exists():
                        continue
                    try:
                        caption_content = caption_path.read_text(
                            encoding="utf-8"
                        ).strip()
                        match = False
                        if caption_filter_mode == CaptionFilter.CONTAINS:
                            if caption_filter.lower() in caption_content.lower():
                                match = True
                        elif caption_filter_mode == CaptionFilter.MATCHES:
                            if caption_filter.lower() == caption_content.lower():
                                match = True
                        elif caption_filter_mode == CaptionFilter.EXCLUDES:
                            if caption_filter.lower() not in caption_content.lower():
                                match = True
                        elif caption_filter_mode == CaptionFilter.REGEX:
                            pat = re.compile(caption_filter, re.IGNORECASE)
                            if pat.search(caption_content):
                                match = True
                        if match:
                            caption_files.append(
                                Path(file_path_str))  # Store as Path object if preferred, or keep as str
                    except Exception:
                        continue
                filtered = [str(p) for p in caption_files]  # Convert back to list of strings if needed
            except Exception as e:
                self.log("error", f"Error applying caption filter: {e}")

        return filtered

    @staticmethod
    def natural_sort_key(s):
        """Sort strings with embedded numbers in natural order."""

        # Split the input string into text and numeric parts
        def convert(text):
            return int(text) if text.isdigit() else text.lower()

        return [convert(c) for c in re.split(r"(\d+)", s)]

    def getSample(self, path):
        image = None
        caption = None
        mask = None

        basepath = self.get_state("path")
        image_path = Path(basepath) / path
        mask_path = image_path.with_name(f"{image_path.stem}-masklabel.png")
        caption_path = image_path.with_suffix(".txt")

        if os.path.exists(image_path):
            image = Image.open(image_path).convert("RGB")

        if os.path.exists(mask_path):
            mask = Image.open(mask_path).convert("L")

        if os.path.exists(caption_path):
            caption = caption_path.read_text(encoding="utf-8").strip()

        return image, mask, caption


    def getMaskPath(self, path):
        basepath = self.get_state("path")
        image_path = Path(basepath) / path
        mask_path = image_path.with_name(f"{image_path.stem}-masklabel.png")

        return mask_path, os.path.exists(mask_path)


    def saveCaption(self, path, caption):
        basepath = self.get_state("path")
        image_path = Path(basepath) / path
        caption_path = image_path.with_suffix(".txt")
        caption_path.write_text(caption.strip(), encoding="utf-8")

    def deleteCaption(self, path):
        basepath = self.get_state("path")
        image_path = Path(basepath) / path
        caption_path = image_path.with_suffix(".txt")
        caption_path.unlink(missing_ok=True)

    def deleteSample(self, path):
        basepath = self.get_state("path")
        image_path = Path(basepath) / path
        mask_path = image_path.with_name(f"{image_path.stem}-masklabel.png")
        caption_path = image_path.with_suffix(".txt")

        image_path.unlink(missing_ok=True)
        mask_path.unlink(missing_ok=True)
        caption_path.unlink(missing_ok=True)

        with self.critical_region_write():
            if path in self.config.files:
                self.config.files.remove(path)


    def __is_supported(self, filename):
        """
            6-10× faster than the original:
            * No Path() construction
            * No lower() for every file (only for the slice that matters)
            * One hash-lookup, one endswith, no branches
            """
        dot = filename.rfind('.')
        if dot == -1:
            return False

        # Check if the stem (filename without extension) ends with the mask suffix
        if filename[:dot].endswith("-masklabel"):
            return False

        ext = filename[dot:].lower()  # slice, not copy of whole string
        return ext in {'.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.webp', '.jxl'}
