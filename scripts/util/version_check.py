#!/usr/bin/env python
from __future__ import print_function
import sys


# Python Version Check.
# IMPORTANT: All code below must be backwards-compatible with Python 2+.


def exit_err(msg):
    sys.stderr.write("Error: " + msg + "\n")
    sys.exit(1)

def str_to_tuple(data):
    try:
        return tuple(map(lambda x: int(x, 10), data.split(".")))
    except Exception:
        exit_err("Invalid version format: " + data)

def tuple_to_str(data):
    return ".".join(map(str, data))

if len(sys.argv) == 1:
    # No arguments: print the version in a controlled format.
    print("Python " + sys.version.split()[0])
    sys.exit(0)
elif len(sys.argv) < 3:
    exit_err("Version check requires 2 arguments: [min_ver] [too_high_ver]")

min_ver = str_to_tuple(sys.argv[1])
too_high_ver = str_to_tuple(sys.argv[2])

if sys.version_info < min_ver:
    exit_err("Your Python version is too low: {}. Must be >= {} and < {}."
             .format(sys.version.split()[0], tuple_to_str(min_ver), tuple_to_str(too_high_ver)))
elif sys.version_info >= too_high_ver:
    exit_err("Your Python version is too high: {}. Must be >= {} and < {}."
             .format(sys.version.split()[0], tuple_to_str(min_ver), tuple_to_str(too_high_ver)))
else:
    # Python version is acceptable; exit silently.
    sys.exit(0)
