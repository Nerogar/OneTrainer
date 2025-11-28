from modules.util.enum.BaseEnum import BaseEnum


class CaptionFilter(BaseEnum):
    CONTAINS = "contains"
    MATCHES = "matches"
    EXCLUDES = "excludes"
    REGEX = "regex"
