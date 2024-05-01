import os
import sys

sys.path.append(os.getcwd())

from modules.zluda import ZLUDAInstaller


if __name__ == '__main__':
    try:
        ZLUDAInstaller.install()
        zluda_path = ZLUDAInstaller.find()
        ZLUDAInstaller.make_copy(zluda_path)
    except Exception as e:
        print(f'Failed to install ZLUDA: {e}')
        sys.exit(1)

    print(f'ZLUDA installed: {zluda_path}')
