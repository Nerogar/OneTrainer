import os
import platform
from modules.zluda.util import find_zluda


if platform.system() == 'Windows':
    zluda_path = find_zluda()
    if os.path.exists(zluda_path):
        paths = os.environ.get('PATH', '.')
        if zluda_path not in paths:
            os.environ['PATH'] = paths + ';' + zluda_path
