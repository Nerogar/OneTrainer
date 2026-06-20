#!/bin/bash
set -o pipefail

if [ -z "$1" ]; then
    echo "Usage: $0 <comfyui-path>" >&2
    exit 1
fi
COMFYUI_PATH="$1"
COMFYUI_PYTHON="$COMFYUI_PATH/venv/bin/python3"

source "$(dirname "$0")/lora_presets_common.sh"

failed=0

for preset in "${presets[@]}"; do
    name=$(preset_name "$preset")
    logfile=$(preset_logfile "$preset")
    lora_path=$(preset_lora_path "$preset")

    if [ ! -f "$lora_path" ]; then
        echo "SKIPPED comfy smoke test for $name (no LoRA found at $lora_path, training did not produce one)" | tee -a "$logfile"
        echo "FAILED (comfy load): $name" | tee -a "$ERROR_LOG"
        failed=1
        continue
    fi

    model_type=$(jq -r .model_type "$preset")
    if printf '%s\n' "${comfy_unsupported[@]}" | grep -qx "$model_type"; then
        echo "SKIPPED comfy smoke test for $name ($model_type not supported by ComfyUI)" | tee -a "$logfile"
        echo "OK (comfy load): $name" | tee -a "$ERROR_LOG"
        continue
    fi

    if ! comfy_files=$(venv/bin/python3 test/download_comfy_smoke_models.py "$model_type" 2>>"$logfile"); then
        echo "FAILED (comfy model download): $name" | tee -a "$ERROR_LOG"
        failed=1
        continue
    fi

    if ! "$COMFYUI_PYTHON" test/smoke_load_comfy.py "$COMFYUI_PATH" "$model_type" "$lora_path" $comfy_files 2>&1 | tee -a "$logfile"; then
        echo "FAILED (comfy load): $name" | tee -a "$ERROR_LOG"
        failed=1
        continue
    fi

    echo "OK (comfy load): $name" | tee -a "$ERROR_LOG"
done

exit $failed
