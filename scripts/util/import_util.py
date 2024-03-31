def script_imports():
    import logging
    import os
    import sys

    # filter out the triton warning on startup
    logging \
        .getLogger("xformers") \
        .addFilter(lambda record: 'A matching Triton is not available' not in record.getMessage())

    sys.path.append(os.getcwd())

    if sys.platform.startswith('win'):
        from modules.zluda.util import find_zluda

        zluda_path = find_zluda()
        if os.path.exists(zluda_path):
            paths = os.environ.get('PATH', '.')
            if zluda_path not in paths:
                os.environ['PATH'] = paths + ';' + zluda_path
