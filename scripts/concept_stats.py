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
                "image_with_mask_count" : "-",
                "image_with_caption_count" : "-",
                "mask_count" : 0,
                "unpaired_masks" : "-",
                "caption_count" : 0,
                "unpaired_captions" : "-",
                "processing_time" : 0,
                "directory_count" : 0,
                "max_pixels" : "-",
                "min_pixels" : "-",
                "avg_pixels" : "-",
                "aspect_buckets" : {}
            }

    #break and return defaults if no path or nonexistent path
    if not os.path.isdir(conceptconfig.path):
        return stats_dict

    #advanced stats default to "-" above if not measured, but need to be initialized if they are
    if advanced_checks:
        stats_dict["image_with_mask_count"] = 0
        stats_dict["image_with_caption_count"] = 0
        stats_dict["unpaired_masks"] = 0
        stats_dict["unpaired_captions"] = 0
        stats_dict["max_pixels"] = [0,"-","-"]
        stats_dict["min_pixels"] = [1000000000,"-","-"]
        stats_dict["avg_pixels"] = 0

        #currently hardcoded, pull from AspectBucketing in mgds eventually
        all_possible_input_aspects = [
                (1.0, 1.0),
                (1.0, 1.25),
                (1.0, 1.5),
                (1.0, 1.75),
                (1.0, 2.0),
                (1.0, 2.5),
                (1.0, 3.0),
                (1.0, 3.5),
                (1.0, 4.0),
            ]

        aspect_ratio_list = []
        for aspect in all_possible_input_aspects:
            aspect_ratio_list.append(round(aspect[0]/aspect[1], 2))
            aspect_ratio_list.append(round(aspect[1]/aspect[0], 2))
        aspect_ratio_list = list(set(aspect_ratio_list))
        aspect_ratio_list.sort()

        #initialize counts for all buckets to 0
        for aspect in aspect_ratio_list:
            stats_dict["aspect_buckets"][aspect] = 0

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

    paired_masks = 0
    paired_captions = 0

    for dir in dir_list:
        file_list = [f for f in os.scandir(dir) if f.is_file()]
        file_list_str = [x.path for x in file_list]     #faster to check list of strings than list of path objects
        for path in file_list:
            basename, extension = os.path.splitext(path)
            if extension.lower() in extensions_list and not path.name.endswith("-masklabel.png"):
                stats_dict["image_count"] += 1
                stats_dict["file_size"] += path.stat().st_size
                if advanced_checks:
                    #check if image has a corresponding mask/caption in the same directory
                    if (basename + "-masklabel.png") in file_list_str:
                        paired_masks += 1
                        stats_dict["image_with_mask_count"] += 1
                    if (basename + ".txt") in file_list_str:
                        paired_captions += 1
                        stats_dict["image_with_caption_count"] += 1
                    #get image resolution info
                    img = Image.open(path)
                    width, height = img.size
                    pixels = width*height
                    true_aspect = width/height
                    nearest_aspect = min(aspect_ratio_list, key=lambda x:abs(x-true_aspect))
                    stats_dict["aspect_buckets"][nearest_aspect] += 1
                    if pixels > stats_dict["max_pixels"][0]:
                        stats_dict["max_pixels"] = [pixels, os.path.relpath(path, conceptconfig.path), f'{width}x{height}']
                    if pixels < stats_dict["min_pixels"][0]:
                        stats_dict["min_pixels"] = [pixels, os.path.relpath(path, conceptconfig.path), f'{width}x{height}']
                    stats_dict["avg_pixels"] += (pixels - stats_dict["avg_pixels"])/stats_dict["image_count"]

            elif path.name.endswith("-masklabel.png"):
                stats_dict["mask_count"] += 1
                stats_dict["file_size"] += path.stat().st_size
            elif extension == ".txt":
                stats_dict["caption_count"] += 1
                stats_dict["file_size"] += path.stat().st_size

        #update every directory loop
        stats_dict["directory_count"] += 1
        if advanced_checks:
            stats_dict["unpaired_masks"] = stats_dict["mask_count"]-paired_masks
            stats_dict["unpaired_captions"] = stats_dict["caption_count"]-paired_captions
        stats_dict["processing_time"] = round(time.perf_counter() - time_start, 3)

    return stats_dict
