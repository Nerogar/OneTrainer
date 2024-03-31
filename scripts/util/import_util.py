def script_imports():
    import logging
    import os
    import sys
    import shutil

    # filter out the triton warning on startup
    logging \
        .getLogger("xformers") \
        .addFilter(lambda record: 'A matching Triton is not available' not in record.getMessage())

    sys.path.append(os.getcwd())

    if sys.platform.startswith('win'):
        from modules.zluda.util import find_zluda

        zluda_path = find_zluda()
        use_zluda = shutil.which('zluda') is not None

        if os.path.exists(zluda_path):
            use_zluda = True
            paths = os.environ.get('PATH', '.')
            if zluda_path not in paths:
                os.environ['PATH'] = zluda_path + ';' + paths

        if use_zluda:
            from modules.zluda import ZLUDA

            ZLUDA.initialize()
