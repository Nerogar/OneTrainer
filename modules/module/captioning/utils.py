import csv
import os
import re
from pathlib import Path

from modules.util import path_util


def is_supported_image(path: Path) -> bool:
    """Returns True if the file has a supported image extension and is not a mask label image."""
    return path_util.is_supported_image_extension(path.suffix) and "-masklabel.png" not in path.name


def get_sample_filenames(sample_dir: str, include_subdirs: bool = False) -> list[str]:
    sample_path = Path(sample_dir)
    pattern = "**/*" if include_subdirs else "*"
    return [str(p) for p in sample_path.glob(pattern) if p.is_file() and is_supported_image(p)]


def is_empty_caption(caption: str) -> bool:
    # Check effectively empty captions.
    # Removes whitespace and punctuation that doesn't affect content.
    stripped = re.sub(r"[\s\.,_`();:'\"-]+", "", caption)
    return not stripped


def get_blacklist_tags(blacklist_text: str, model_name: str) -> list[str]:
    """Convert blacklist_text to list depending on whether it's a file or a comma-separated string."""
    delimiter = "," if "WD" in model_name or "," in blacklist_text else None

    if blacklist_text.endswith(".txt") and os.path.isfile(blacklist_text):
        with open(blacklist_text, encoding="utf-8") as blacklist_file:
            return [line.rstrip("\n") for line in blacklist_file]
    elif blacklist_text.endswith(".csv") and os.path.isfile(blacklist_text):
        with open(blacklist_text, "r", encoding="utf-8") as blacklist_file:
            return [row[0] for row in csv.reader(blacklist_file)]
    elif delimiter:
        return [tag.strip() for tag in blacklist_text.split(delimiter)]
    else:
        return [blacklist_text]


def parse_regex_blacklist(blacklist_tags: list[str], caption_tags: list[str]) -> list[str]:
    """Match regex patterns in blacklist against caption tags."""
    matched_tags: list[str] = []
    regex_spchars = set(".^$*+?!{}[]|()\\")
    for pattern in blacklist_tags:
        if any(char in regex_spchars for char in pattern):
            try:
                compiled = re.compile(pattern, re.IGNORECASE)
                for tag in caption_tags:
                    if compiled.fullmatch(tag) and tag not in matched_tags:
                        matched_tags.append(tag)
            except re.error:
                pass
        else:
            pattern_lower = pattern.lower()
            for tag in caption_tags:
                if tag.lower() == pattern_lower and tag not in matched_tags:
                    matched_tags.append(tag)
    return matched_tags


def filter_blacklisted_tags(caption: str, blacklist_text: str, model_name: str, use_regex: bool = False) -> str:
    """Remove blacklisted tags from a caption."""
    if not blacklist_text:
        return caption

    # Check if the model contains "WD" and choose delimiter and joiner accordingly.
    if "WD" in model_name:
        delimiter = ","
        joiner = ", "
    else:
        delimiter = " "
        joiner = " "

    blacklist_tags = get_blacklist_tags(blacklist_text, model_name)
    # Trim each tag from the caption by splitting on the chosen delimiter.
    caption_tags = [tag.strip() for tag in caption.split(delimiter)]

    if use_regex:
        tags_to_remove = parse_regex_blacklist(blacklist_tags, caption_tags)
    else:
        tags_to_remove = []
        for tag in caption_tags:
            # Strip punctuation for comparison but keep original tag for removal
            tag_cleaned = re.sub(r'[^\w\s]', '', tag.lower())
            for blacklist_tag in blacklist_tags:
                blacklist_cleaned = re.sub(r'[^\w\s]', '', blacklist_tag.lower().strip())
                if tag_cleaned == blacklist_cleaned:
                    tags_to_remove.append(tag)
                    break

    filtered_tags = [tag for tag in caption_tags if tag not in tags_to_remove]
    return joiner.join(filtered_tags)
