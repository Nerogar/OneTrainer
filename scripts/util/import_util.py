def script_imports():
    import logging
    import os
    import sys

    # filter out the triton warning on startup
    logging \
        .getLogger("xformers") \
        .addFilter(lambda record: 'A matching Triton is not available' not in record.getMessage())

    sys.path.append(os.getcwd())
