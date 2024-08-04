import os
import ctypes
import shutil
import zipfile
import platform
import urllib.request
import re
from typing import Union


RELEASE = f"rel.{os.environ.get('ZLUDA_HASH', '86cdab3b14b556e95eafe370b8e8a1a80e8d093b')}"
DLL_MAPPING = {
    'cublas.dll': 'cublas64_11.dll',
    'cusparse.dll': 'cusparse64_11.dll',
    'nvrtc.dll': 'nvrtc64_112_0.dll',
}
HIP_TARGETS_COMMON = ['rocblas.dll', 'rocsolver.dll']
ZLUDA_TARGETS = ('nvcuda.dll', 'nvml.dll')


def get_path() -> str:
    return os.path.abspath(os.environ.get('ZLUDA', '.zluda'))


def find_hip_sdk() -> Union[str, None]:
    program_files = os.environ.get('ProgramFiles', r'C:\Program Files')
    hip_path_default = rf'{program_files}\AMD\ROCm\5.7'
    if not os.path.exists(hip_path_default):
        hip_path_default = None
    return os.environ.get('HIP_PATH', hip_path_default)


def install(zluda_path: os.PathLike) -> None:
    if os.path.exists(zluda_path):
        return

    if platform.system() != 'Windows': # TODO
        return

    urllib.request.urlretrieve(f'https://github.com/lshqqytiger/ZLUDA/releases/download/{RELEASE}/ZLUDA-windows-amd64.zip', '_zluda')
    with zipfile.ZipFile('_zluda', 'r') as archive:
        infos = archive.infolist()
        for info in infos:
            if not info.is_dir():
                info.filename = os.path.basename(info.filename)
                archive.extract(info, '.zluda')
    os.remove('_zluda')


def make_copy(zluda_path: os.PathLike) -> None:
    for k, v in DLL_MAPPING.items():
        if not os.path.exists(os.path.join(zluda_path, v)):
            try:
                os.link(os.path.join(zluda_path, k), os.path.join(zluda_path, v))
            except Exception:
                shutil.copyfile(os.path.join(zluda_path, k), os.path.join(zluda_path, v))


def find_hiprtc(hip_path: os.PathLike) -> str:
    hip_bin_path = os.path.join(hip_path, 'bin')
    for v in os.listdir(hip_bin_path):
        if re.match(r'hiprtc\d+\.dll', v):
            return os.path.join(hip_bin_path, v)


def load(zluda_path: os.PathLike) -> None:
    hip_path = find_hip_sdk()
    if hip_path is None:
        raise RuntimeError('Could not find AMD HIP SDK, please install it from https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html')

    hiptrc_path = find_hiprtc(hip_path)
    if hiptrc_path:
        ctypes.windll.LoadLibrary(hiptrc_path)
    else:
        raise RuntimeError('Could not find hiprtc*.dll in the AMD HIP SDK')

    for v in HIP_TARGETS_COMMON:
        ctypes.windll.LoadLibrary(os.path.join(hip_path, 'bin', v))
    for v in ZLUDA_TARGETS:
        ctypes.windll.LoadLibrary(os.path.join(zluda_path, v))
    for v in DLL_MAPPING.values():
        ctypes.windll.LoadLibrary(os.path.join(zluda_path, v))
