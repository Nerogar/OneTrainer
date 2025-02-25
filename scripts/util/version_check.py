import sys


# Python Version Check.
# IMPORTANT: All code below must be backwards-compatible with Python 2+.


def exit_err(msg):
    sys.stderr.write("Error: " + msg + "\n")
    sys.stderr.flush()
    sys.exit(1)


def str_to_tuple(data):
    return tuple(map(lambda x: int(x, 10), data.split(".")))


def tuple_to_str(data):
    return ".".join(map(str, data))


def exit_wrong_version(msg, min_ver, too_high_ver):
    exit_err(
        "Your Python version is %s: %s. Must be >= %s and < %s."
        % (msg, sys.version, tuple_to_str(min_ver), tuple_to_str(too_high_ver))
    )


if len(sys.argv) < 3:
    exit_err("Version check requires 2 arguments: [min_ver] [too_high_ver]")

min_ver = str_to_tuple(sys.argv[1])
too_high_ver = str_to_tuple(sys.argv[2])

if sys.version_info < min_ver:
    exit_wrong_version("too low", min_ver, too_high_ver)

if sys.version_info >= too_high_ver:
    exit_wrong_version("too high", min_ver, too_high_ver)
