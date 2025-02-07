from util.import_util import script_imports

script_imports()

import os
import time

from modules.util import path_util
from modules.util.config.ConceptConfig import ConceptConfig

from PIL import Image


def get_concept_stats(conceptconfig : ConceptConfig, advanced_checks : bool):
    stats_dict = {
                "file_size" : 0,
                "image_count" : 0,
                "image_with_mask_count" : 0,
                "image_with_caption_count" : 0,
                "mask_count" : 0,
                "paired_masks" : 0,
                "caption_count" : 0,
                "paired_captions" : 0,
                "processing_time" : 0,
                "directory_count" : 0,
                "max_pixels" : 0,
                "min_pixels" : 1000000000,
                "avg_pixels" : 0
            }

    #fast recursive directory scan from https://stackoverflow.com/a/40347279
    def fast_scandir(dirname):
        subfolders = [f for f in os.scandir(dirname) if f.is_dir()]
        for dirname in subfolders:
            subfolders.extend(fast_scandir(dirname))
        return subfolders

    time_start = time.perf_counter()
    extensions_list = path_util.SUPPORTED_IMAGE_EXTENSIONS
    if conceptconfig.include_subdirectories:
        dir_list = fast_scandir(conceptconfig.path)
        dir_list.append(conceptconfig.path)      #add top-level directory as well
    else:
        dir_list = [conceptconfig.path]

    for dir in dir_list:
        file_list = [f for f in os.scandir(dir) if f.is_file]
        file_list_str = [x.path for x in file_list]
        for path in file_list:
            basename, extension = os.path.splitext(path)
            if extension.lower() in extensions_list and not path.name.endswith("-masklabel.png"):
                stats_dict["image_count"] += 1
                stats_dict["file_size"] += path.stat().st_size
                if advanced_checks:
                    #check if image has a corresponding mask/caption in the same directory
                    if (basename + "-masklabel.png") in file_list_str:
                        stats_dict["paired_masks"] += 1
                        stats_dict["image_with_mask_count"] += 1
                    if (basename + ".txt") in file_list_str:
                        stats_dict["paired_captions"] += 1
                        stats_dict["image_with_caption_count"] += 1
                    #get image resolution info
                    img = Image.open(path)
                    width, height = img.size
                    pixels = width*height
                    stats_dict["max_pixels"] = max(pixels, stats_dict["max_pixels"])
                    stats_dict["min_pixels"] = min(pixels, stats_dict["min_pixels"])
                    stats_dict["avg_pixels"] += (pixels - stats_dict["avg_pixels"])/stats_dict["image_count"]

            elif path.name.endswith("-masklabel.png"):
                stats_dict["mask_count"] += 1
                stats_dict["file_size"] += path.stat().st_size
            elif extension == ".txt":
                stats_dict["caption_count"] += 1
                stats_dict["file_size"] += path.stat().st_size

    stats_dict["directory_count"] = len(dir_list)
    stats_dict["processing_time"] = round(time.perf_counter() - time_start, 3)

    return stats_dict
