import logging
import os


def silence_pyvips(enable_silence: bool = True) -> None:
    """
    Silence PyVips logging and console output.

    Args:
        enable_silence: If True (default), silence PyVips output. If False, restore default logging.
    """
    # Set logging level for PyVips
    level = logging.WARNING if enable_silence else logging.INFO
    logging.getLogger('pyvips').setLevel(level)

    # Control C library output through environment variables
    if enable_silence:
        os.environ['VIPS_SILENT'] = '1'        # Suppress most messages
        os.environ['VIPS_PROGRESS'] = '0'      # Disable progress updates
        os.environ['VIPS_VERBOSITY'] = '0'     # Minimal verbosity
    else:
        # Restore default behavior
        os.environ.pop('VIPS_SILENT', None)
        os.environ.pop('VIPS_PROGRESS', None)
        os.environ.pop('VIPS_VERBOSITY', None)

def set_log_level(module_name: str, level: int | str) -> None:
    """
    Set the logging level for a specific module.

    Args:
        module_name: Name of the module to configure
        level: Logging level (can be int constant like logging.WARNING or string like 'WARNING')
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    logging.getLogger(module_name).setLevel(level)
