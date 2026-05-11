# OneTrainer dual-GPU FLUX.2 LoRA training smoke test.
#
# 10-step run on sumi v8 dataset, FLUX2_DUAL_GPU=true.

$ErrorActionPreference = "Stop"
$env:FLUX2_DUAL_GPU = "true"
$env:PYTHONIOENCODING = "utf-8"

Set-Location C:\OneTrainer

$python = "C:\OneTrainer\.venv\Scripts\python.exe"

& $python scripts/train.py --config-path C:\OneTrainer\configs\sumi_dualgpu.json
