import concurrent.futures
import os
import threading
import time

from modules.util import path_util
from modules.util.config.ConceptConfig import ConceptConfig
from modules.util.image_util import load_image

from mgds.pipelineModules.AspectBucketing import AspectBucketing

import cv2
import imagesize


def init_concept_stats(advanced_checks : bool):
    stats_dict = {
                "file_size" : 0,
                "image_count" : 0,
                "image_with_mask_count" : "-",
                "image_with_caption_count" : "-",
                "video_count" : 0,
                "video_with_mask_count" : "-",
                "video_with_caption_count" : "-",
                "mask_count" : 0,
                "paired_masks" : "-",
                "unpaired_masks" : "-",
                "caption_count" : 0,
                "subcaption_count" : 0,
                "paired_captions" : "-",
                "unpaired_captions" : "-",
                "processing_time" : 0,
                "directory_count" : 0,
                "max_pixels" : "-",
                "min_pixels" : "-",
                "avg_pixels" : "-",
                "max_length" : "-",
                "min_length" : "-",
                "avg_length" : "-",
                "max_fps" : "-",
                "min_fps" : "-",
                "avg_fps" : "-",
                "max_caption_length" : "-",
                "min_caption_length" : "-",
                "avg_caption_length" : "-",
                "aspect_buckets" : {},
                "force_cancelled" : False
            }

    #advanced stats default to "-" above if not measured, but need to be initialized to specific values if they are
    if advanced_checks:
        stats_dict["image_with_mask_count"] = 0
        stats_dict["image_with_caption_count"] = 0
        stats_dict["video_with_mask_count"] = 0
        stats_dict["video_with_caption_count"] = 0
        stats_dict["paired_masks"] = 0
        stats_dict["unpaired_masks"] = 0
        stats_dict["paired_captions"] = 0
        stats_dict["unpaired_captions"] = 0
        stats_dict["max_pixels"] = [0,"-","-"]                  #max pixels, file path, resolution (wxh)
        stats_dict["min_pixels"] = [1000000000,"-","-"]         #min pixels, file path, resolution (wxh)
        stats_dict["avg_pixels"] = 0
        stats_dict["max_length"] = [0,"-"]                      #max frame count, file path
        stats_dict["min_length"] = [1000000000,"-"]             #min frame count, file path
        stats_dict["avg_length"] = 0
        stats_dict["max_fps"] = [0,"-"]                         #max fps file path
        stats_dict["min_fps"] = [1000000000,"-"]                #min fps, file path
        stats_dict["avg_fps"] = 0
        stats_dict["max_caption_length"] = [0,"-",0]            #max char count, filepath, word count
        stats_dict["min_caption_length"] = [1000000000,"-",0]   #min char count, filepath, word count
        stats_dict["avg_caption_length"] = [0,0]                #avg char count, avg word count

        aspect_ratio_list = []
        for aspect in AspectBucketing.all_possible_input_aspects:   #input parameters don't matter but can't be blank
            aspect_ratio_list.append(round(aspect[0]/aspect[1], 6))     #get both wide and tall ratios
            aspect_ratio_list.append(round(aspect[1]/aspect[0], 6))
        aspect_ratio_list = list(set(aspect_ratio_list))
        aspect_ratio_list.sort()

        #initialize counts for all buckets to 0
        for aspect in aspect_ratio_list:
            stats_dict["aspect_buckets"][aspect] = 0

    return stats_dict

def folder_scan(dir, stats_dict : dict, advanced_checks : bool, conceptconfig : ConceptConfig, start_time : float, wait_time : float, cancel_scan_flag : threading.Event):
    #break and return defaults if no path or nonexistent path
    if not os.path.isdir(dir):
        stats_dict["force_cancelled"] = True
        return stats_dict
    img_extensions_list = path_util.SUPPORTED_IMAGE_EXTENSIONS
    vid_extensions_list = path_util.SUPPORTED_VIDEO_EXTENSIONS
    file_list = [f for f in os.scandir(dir) if f.is_file()]     #this may take time on large directories
    if advanced_checks:
        aspect_ratio_list = list(stats_dict["aspect_buckets"].keys())
        file_list_str = [x.path for x in file_list]     #seems faster to check list of strings for matching files than list of path objects
    stats_dict["directory_count"] += 1

    for path in file_list:
        stats_dict["processing_time"] = time.perf_counter() - start_time
        if time.perf_counter() - start_time > wait_time or cancel_scan_flag.is_set():
            stats_dict["force_cancelled"] = True
            return stats_dict
        basename, extension = os.path.splitext(path)
        if extension.lower() in img_extensions_list and not path.name.endswith("-masklabel.png") and not path.name.endswith("-condlabel.png"):
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
                        #get character/word count of captions, split by newlines in each text file
                        for caption in captionlist:
                            stats_dict["subcaption_count"] += 1     #each line in one file
                            char_count = len(caption)
                            word_count = len(caption.split())
                            if char_count > stats_dict["max_caption_length"][0]:
                                stats_dict["max_caption_length"] = [char_count, os.path.relpath(path, conceptconfig.path), word_count]
                            if char_count < stats_dict["min_caption_length"][0]:
                                stats_dict["min_caption_length"] = [char_count, os.path.relpath(path, conceptconfig.path), word_count]
                            stats_dict["avg_caption_length"][0] += (char_count - stats_dict["avg_caption_length"][0])/(stats_dict["image_count"] + stats_dict["video_count"])
                            stats_dict["avg_caption_length"][1] += (word_count - stats_dict["avg_caption_length"][1])/(stats_dict["image_count"] + stats_dict["video_count"])

                #get image resolution info
                try:    #use imagesize if possible due to better speed
                    width, height = imagesize.get(path.path)
                    if width == -1:     #if imagesize doesn't recognize format it returns (-1, -1)
                        raise ValueError
                except ValueError:     #use PIL if not supported by imagesize
                    img = load_image(path.path)
                    width, height = img.size
                    img.close()
                pixels = width*height
                true_aspect = height/width
                nearest_aspect = min(aspect_ratio_list, key=lambda x:abs(x-true_aspect))    #try to match math used in aspect bucketing
                stats_dict["aspect_buckets"][nearest_aspect] += 1

                if pixels > stats_dict["max_pixels"][0]:
                    stats_dict["max_pixels"] = [pixels, os.path.relpath(path, conceptconfig.path), f'{width}w x {height}h']
                if pixels < stats_dict["min_pixels"][0]:
                    stats_dict["min_pixels"] = [pixels, os.path.relpath(path, conceptconfig.path), f'{width}w x {height}h']
                stats_dict["avg_pixels"] += (pixels - stats_dict["avg_pixels"])/(stats_dict["image_count"] + stats_dict["video_count"])

        elif extension.lower() in vid_extensions_list:
            stats_dict["video_count"] += 1
            stats_dict["file_size"] += path.stat().st_size
            if advanced_checks:
                #check if video has a corresponding mask/caption in the same directory
                # if (basename + "-masklabel.png") in file_list_str:
                #     stats_dict["paired_masks"] += 1
                #     stats_dict["video_with_mask_count"] += 1
                if (basename + ".txt") in file_list_str:
                    stats_dict["paired_captions"] += 1
                    stats_dict["video_with_caption_count"] += 1
                    with open(basename + ".txt", "r") as captionfile:
                        captionlist = captionfile.read().splitlines()
                        #get character/word count of captions, split by newlines in each text file
                        for caption in captionlist:
                            stats_dict["subcaption_count"] += 1     #each line in one file
                            char_count = len(caption)
                            word_count = len(caption.split())
                            if char_count > stats_dict["max_caption_length"][0]:
                                stats_dict["max_caption_length"] = [char_count, os.path.relpath(path, conceptconfig.path), word_count]
                            if char_count < stats_dict["min_caption_length"][0]:
                                stats_dict["min_caption_length"] = [char_count, os.path.relpath(path, conceptconfig.path), word_count]
                            stats_dict["avg_caption_length"][0] += (char_count - stats_dict["avg_caption_length"][0])/(stats_dict["image_count"] + stats_dict["video_count"])
                            stats_dict["avg_caption_length"][1] += (word_count - stats_dict["avg_caption_length"][1])/(stats_dict["image_count"] + stats_dict["video_count"])

                vid = cv2.VideoCapture(path)
                width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
                length = vid.get(cv2.CAP_PROP_FRAME_COUNT)
                fps = vid.get(cv2.CAP_PROP_FPS)
                vid.release()

                pixels = width*height
                true_aspect = height/width
                nearest_aspect = min(aspect_ratio_list, key=lambda x:abs(x-true_aspect))
                stats_dict["aspect_buckets"][nearest_aspect] += 1

                if pixels > stats_dict["max_pixels"][0]:
                    stats_dict["max_pixels"] = [pixels, os.path.relpath(path, conceptconfig.path), f'{width}w x {height}h']
                if pixels < stats_dict["min_pixels"][0]:
                    stats_dict["min_pixels"] = [pixels, os.path.relpath(path, conceptconfig.path), f'{width}w x {height}h']
                stats_dict["avg_pixels"] += (pixels - stats_dict["avg_pixels"])/(stats_dict["image_count"] + stats_dict["video_count"])

                if length > stats_dict["max_length"][0]:
                    stats_dict["max_length"] = [length, os.path.relpath(path, conceptconfig.path)]
                if length < stats_dict["min_length"][0]:
                    stats_dict["min_length"] = [length, os.path.relpath(path, conceptconfig.path)]
                stats_dict["avg_length"] += (length - stats_dict["avg_length"])/stats_dict["video_count"]

                if fps > stats_dict["max_fps"][0]:
                    stats_dict["max_fps"] = [fps, os.path.relpath(path, conceptconfig.path)]
                if fps < stats_dict["min_fps"][0]:
                    stats_dict["min_fps"] = [fps, os.path.relpath(path, conceptconfig.path)]
                stats_dict["avg_fps"] += (fps - stats_dict["avg_fps"])/stats_dict["video_count"]

        elif path.name.endswith("-masklabel.png"):
            stats_dict["mask_count"] += 1
            stats_dict["file_size"] += path.stat().st_size
        elif extension == ".txt":
            stats_dict["caption_count"] += 1
            stats_dict["file_size"] += path.stat().st_size

    #update every directory loop
    if advanced_checks:
        #check for number of "orphaned" mask/caption files as the difference between the total count and the count of image/mask or image/caption pairs
        stats_dict["unpaired_masks"] = stats_dict["mask_count"]-stats_dict["paired_masks"]
        stats_dict["unpaired_captions"] = stats_dict["caption_count"]-stats_dict["paired_captions"]

    return stats_dict

#currently unused
def combine_stats_dicts(input_dicts : list[dict], advanced_checks : bool):
    final_dict = init_concept_stats(advanced_checks)
    total_pixels = 0
    total_fps = 0
    total_length = 0
    total_caption_length = [0,0]

    for dict in input_dicts:
        if dict["force_cancelled"]:     #if any scans were cancelled indicate it on the final result
            final_dict["force_cancelled"] = True
            continue
        for key in dict:
            if key in ["file_size", "image_count", "video_count",
                "mask_count", "caption_count", "directory_count"] \
                or advanced_checks and key in ["image_with_mask_count", "image_with_caption_count", "video_with_mask_count","subcaption_count",
                "video_with_caption_count", "paired_masks", "unpaired_masks", "paired_captions", "unpaired_captions"]:
                final_dict[key] += dict[key]
            elif advanced_checks and key in ["aspect_buckets"]:
                for subkey in dict[key]:
                    final_dict[key][subkey] += dict[key][subkey]
            elif advanced_checks and key in ["max_pixels", "max_length", "max_fps", "max_caption_length"]:
                if dict[key][0] > final_dict[key][0]:
                    final_dict[key] = dict[key]
            elif advanced_checks and key in ["min_pixels", "min_length", "min_fps", "min_caption_length"]:
                if dict[key][0] < final_dict[key][0]:
                    final_dict[key] = dict[key]
            elif advanced_checks and key in ["avg_pixels"]:     #add averages * count initially to later divide out count
                total_pixels += dict[key] * (dict["image_count"] + dict["video_count"])
            elif advanced_checks and key in ["avg_fps"]:
                total_fps += dict[key] * dict["video_count"]
            elif advanced_checks and key in ["avg_length"]:
                total_length += dict[key] * dict["video_count"]
            elif advanced_checks and key in ["avg_caption_length"]:
                total_caption_length[0] += dict[key][0] * dict["subcaption_count"]
                total_caption_length[1] += dict[key][1] * dict["subcaption_count"]

    if advanced_checks:     #divide out total counts for pixels/length/etc by total image/video count
        try:
            final_dict["avg_pixels"] = total_pixels / (final_dict["image_count"] + final_dict["video_count"])
        except ZeroDivisionError:
            final_dict["avg_pixels"] = 0
        try:
            final_dict["avg_caption_length"][0] = total_caption_length[0] / final_dict["subcaption_count"]
            final_dict["avg_caption_length"][1] = total_caption_length[1] / final_dict["subcaption_count"]
        except ZeroDivisionError:
            final_dict["avg_caption_length"][0] = 0
            final_dict["avg_caption_length"][1] = 0
        try:
            final_dict["avg_fps"] = total_fps / final_dict["video_count"]
            final_dict["avg_length"] = total_length / final_dict["video_count"]
        except ZeroDivisionError:
            final_dict["avg_fps"] = 0
            final_dict["avg_length"] = 0

    return final_dict

#loop through all subfolders of top-level path
#currently unused, but similar logic in ConceptWindow
def subfolder_scan(conceptconfig : ConceptConfig, advanced_checks : bool, wait_time : float):
    stats_dict = init_concept_stats(advanced_checks)
    start_time = time.perf_counter()
    subfolders = [conceptconfig.path]
    cancel_scan_flag = threading.Event()
    cancel_scan_flag.clear()

    for dir in subfolders:
        stats_dict = folder_scan(dir, stats_dict, advanced_checks, start_time, wait_time, cancel_scan_flag)
        subfolders.extend([f for f in os.scandir(dir) if f.is_dir()])

    return stats_dict

#loop through all subfolders of top-level path, uses concurrent.futures to process multiple directories in parallel
#faster for files on HDDs but slower on SSDs, likely due to creating/destroying threads
#currently unused
def subfolder_scan_threaded(conceptconfig : ConceptConfig, advanced_checks : bool, wait_time : float):
    start_time = time.perf_counter()
    subfolders = [conceptconfig.path]
    cancel_scan_flag = threading.Event()
    cancel_scan_flag.clear()

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        stats_futures = []
        for dir in subfolders:
            stats_futures += [executor.submit(folder_scan, dir, init_concept_stats(advanced_checks), advanced_checks, start_time, wait_time, cancel_scan_flag)]
            subfolders.extend([f.path for f in os.scandir(dir) if f.is_dir()])
        stats_results = [f.result() for f in stats_futures]

    final_stats = combine_stats_dicts(stats_results, conceptconfig, advanced_checks)
    final_stats["processing_time"] = time.perf_counter() - start_time
    return final_stats
