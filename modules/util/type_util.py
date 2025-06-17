from typing import get_origin


def issubclass_safe(x, t):
    # if x is defined as a generic list or dict (e.g. `list[int]`), issubclass will throw an error
    return get_origin(x) is not list and get_origin(x) is not dict and issubclass(x, t)
