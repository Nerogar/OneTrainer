"""
The `version_check.py` script is a utility for ensuring the correct Python version.

This script checks that the python version is within a range.
It uses `sys.argv` to check the command line arguments.
It is used at the beginning of the scripts to ensure they are running on the correct version of python.
"""
import sys


# Python Version Check.
# IMPORTANT: All code below must be backwards-compatible with Python 2+.


def exit_err(msg):
    """
    Exits with an error message.

    Prints an error message to the console and exits with code 1.

    Args:
        msg (str): The error message to print.
    """
    print("Error: " + msg)
    exit(1)


def str_to_tuple(data):
    """
    Converts a string representation of a version to a tuple of integers.

    Args:
        data (str): The string representation of the version.

    Returns:
        tuple: A tuple of integers representing the version.
    """
    return tuple(map(lambda x: int(x, 10), data.split(".")))


def tuple_to_str(data):
    """
    Converts a tuple of integers to a string representation.

    Args:
        data (tuple): A tuple of integers.

    Returns:
        str: A string representation of the tuple.
    """
    return ".".join(map(str, data))


def exit_wrong_version(msg, min_ver, too_high_ver):
    """
    Exits with a version error message.

    Prints an error message to the console and exits with code 1.

    Args:
        msg (str): The error message to print.
        min_ver (tuple): The minimum version allowed.
        too_high_ver (tuple): The maximum version allowed.
    """
    exit_err(
        "Your Python version is {}: {}. Must be >= {} and < {}.".format(
            msg, sys.version, tuple_to_str(min_ver), tuple_to_str(too_high_ver)
        )
    )


"""
Checks the Python version against the minimum and maximum allowed versions.

Exits if the number of command line arguments is incorrect.
Converts the minimum and maximum version strings to tuples.
Exits if the python version is too low or too high.
"""
if len(sys.argv) < 3:
    exit_err("Version check requires 2 arguments: [min_ver] [too_high_ver]")

min_ver = str_to_tuple(sys.argv[1])
too_high_ver = str_to_tuple(sys.argv[2])

if sys.version_info < min_ver:
    exit_wrong_version("too low", min_ver, too_high_ver)

if sys.version_info >= too_high_ver:
    exit_wrong_version("too high", min_ver, too_high_ver)
