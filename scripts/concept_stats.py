from util.import_util import script_imports

script_imports()

import os
import time

from modules.util import path_util
from modules.util.config.ConceptConfig import ConceptConfig

from mgds.pipelineModules.AspectBucketing import AspectBucketing

import imagesize
from PIL import Image


def init_concept_stats(conceptconfig : ConceptConfig, advanced_checks : bool):
    stats_dict = {
                "file_size" : 0,
                "image_count" : 0,
                "image_with_mask_count" : "-",
                "image_with_caption_count" : "-",
                "mask_count" : 0,
                "paired_masks" : "-",
                "unpaired_masks" : "-",
                "caption_count" : 0,
                "paired_captions" : "-",
                "unpaired_captions" : "-",
                "processing_time" : 0,
                "directory_count" : 0,
                "max_pixels" : "-",
                "min_pixels" : "-",
                "avg_pixels" : "-",
                "max_caption_length" : "-",
                "min_caption_length" : "-",
                "avg_caption_length" : "-",
                "aspect_buckets" : {}
            }

    #break and return defaults if no path or nonexistent path
    if not os.path.isdir(conceptconfig.path):
        return stats_dict

    #advanced stats default to "-" above if not measured, but need to be initialized if they are
    if advanced_checks:
        stats_dict["image_with_mask_count"] = 0
        stats_dict["image_with_caption_count"] = 0
        stats_dict["paired_masks"] = 0
        stats_dict["unpaired_masks"] = 0
        stats_dict["paired_captions"] = 0
        stats_dict["unpaired_captions"] = 0
        stats_dict["max_pixels"] = [0,"-","-"]                  #max pixels, file path, resolution (wxh)
        stats_dict["min_pixels"] = [1000000000,"-","-"]         #min pixels, file path, resolution (wxh)
        stats_dict["avg_pixels"] = 0
        stats_dict["max_caption_length"] = [0,"-",0]            #max char count, filepath, word count
        stats_dict["min_caption_length"] = [1000000000,"-",0]   #min char count, filepath, word count
        stats_dict["avg_caption_length"] = [0,0]                #avg char count, avg word count

        aspect_ratio_list = []
        for aspect in AspectBucketing(0,"","","","","","","","","").all_possible_input_aspects:   #input parameters don't matter but can't be blank
            aspect_ratio_list.append(round(aspect[0]/aspect[1], 2))
            aspect_ratio_list.append(round(aspect[1]/aspect[0], 2))
        aspect_ratio_list = list(set(aspect_ratio_list))
        aspect_ratio_list.sort()

        #initialize counts for all buckets to 0
        for aspect in aspect_ratio_list:
            stats_dict["aspect_buckets"][aspect] = 0

    return stats_dict

def folder_scan(dir, stats_dict : dict, advanced_checks : bool, conceptconfig : ConceptConfig):
    extensions_list = path_util.SUPPORTED_IMAGE_EXTENSIONS
    aspect_ratio_list = list(stats_dict["aspect_buckets"].keys())
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
                    stats_dict["paired_masks"] += 1
                    stats_dict["image_with_mask_count"] += 1
                if (basename + ".txt") in file_list_str:
                    stats_dict["paired_captions"] += 1
                    stats_dict["image_with_caption_count"] += 1
                    with open(basename + ".txt", "r") as captionfile:
                        captionlist = captionfile.read().splitlines()
                        #get character/word count of captions, split by newlines
                        for caption in captionlist:
                            char_count = len(caption)
                            word_count = len(caption.split())
                            if char_count > stats_dict["max_caption_length"][0]:
                                stats_dict["max_caption_length"] = [char_count, os.path.relpath(path, conceptconfig.path), word_count]
                            if char_count < stats_dict["min_caption_length"][0]:
                                stats_dict["min_caption_length"] = [char_count, os.path.relpath(path, conceptconfig.path), word_count]
                            stats_dict["avg_caption_length"][0] += (char_count - stats_dict["avg_caption_length"][0])/stats_dict["image_count"]
                            stats_dict["avg_caption_length"][1] += (word_count - stats_dict["avg_caption_length"][1])/stats_dict["image_count"]
                #get image resolution info
                try:    #use imagesize if possible due to better speed
                    width, height = imagesize.get(path.path)
                except ValueError:     #use PIL if not supported by imagesize
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
        stats_dict["unpaired_masks"] = stats_dict["mask_count"]-stats_dict["paired_masks"]
        stats_dict["unpaired_captions"] = stats_dict["caption_count"]-stats_dict["paired_captions"]

    return stats_dict

#loop through all subfolders of top-level path
def subfolder_scan(dirpath, stats_dict, advanced_checks):
    start_time = time.perf_counter()
    subfolders = [dirpath]
    for dir in subfolders:
        stats_dict = folder_scan(dir, stats_dict, advanced_checks)
        stats_dict["processing_time"] = time.perf_counter() - start_time
        subfolders.extend([f for f in os.scandir(dir) if f.is_dir()])
    return stats_dict
