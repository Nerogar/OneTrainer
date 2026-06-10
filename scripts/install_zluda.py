from util.import_util import script_imports

script_imports(allow_zluda=False)

import sys

from modules.zluda import ZLUDAInstaller

if __name__ == '__main__':
    try:
        zluda_path = ZLUDAInstaller.get_path()
        ZLUDAInstaller.install(zluda_path)
        ZLUDAInstaller.make_copy(zluda_path)
    except Exception as e:
        print(f'Failed to install ZLUDA: {e}')
        sys.exit(1)

    print(f'ZLUDA installed: {zluda_path}')
