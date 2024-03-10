import os
import shutil
import platform


def find_zluda():
    return os.environ.get('ZLUDA', './.zluda')


def patch_zluda(zluda_path: os.PathLike):
    venv_dir = os.environ.get('VENV_DIR', os.path.dirname(shutil.which('python')))
    dlls_to_patch = {
        'cublas.dll': 'cublas64_11.dll',
        #'cudnn.dll': 'cudnn64_8.dll',
        'cusparse.dll': 'cusparse64_11.dll',
        'nvrtc.dll': 'nvrtc64_112_0.dll',
    }
    try:
        for k, v in dlls_to_patch.items():
            shutil.copyfile(os.path.join(zluda_path, k), os.path.join(venv_dir, 'Lib', 'site-packages', 'torch', 'lib', v))
    except Exception as e:
        print(f'Failed to automatically patch torch: {e}')
        return False
    return True


def fix_path():
    if platform.system() == 'Windows':
        zluda_path = find_zluda()
        paths = os.environ.get('PATH', '.')
        if zluda_path not in paths:
            os.environ['PATH'] = paths + ';' + zluda_path
