import os
import sys

sys.path.append(os.getcwd())

import platform


from modules.zluda.util import patch_zluda


if __name__ == '__main__':
    is_windows = platform.system() == 'Windows'
    zluda_path = os.environ.get('ZLUDA', None)

    if zluda_path is None:
        paths = os.environ.get('PATH', '').split(';')
        for path in paths:
            if os.path.exists(os.path.join(path, 'zluda_redirect.dll')):
                zluda_path = path
                break

    if zluda_path is None:
        import urllib.request
        if is_windows:
            import zipfile
            archive_type = zipfile.ZipFile
            zluda_url = 'https://github.com/lshqqytiger/ZLUDA/releases/download/v3.5-win/ZLUDA-windows-amd64.zip'
        else:
            import tarfile
            archive_type = tarfile.TarFile
            zluda_url = 'https://github.com/vosen/ZLUDA/releases/download/v3/zluda-3-linux.tar.gz'
        try:
            urllib.request.urlretrieve(zluda_url, '_zluda')
            with archive_type('_zluda', 'r') as f:
                f.extractall('.zluda')
            zluda_path = os.path.abspath('./.zluda')
            os.remove('_zluda')
        except Exception as e:
            print(f'Failed to install ZLUDA: {e}')
            exit(1)

    if zluda_path is None:
        exit(1)

    if patch_zluda(zluda_path):
        print(f'ZLUDA installed: {zluda_path}')
