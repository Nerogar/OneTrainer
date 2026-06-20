#!/bin/bash
set -o pipefail

source "$(dirname "$0")/lora_presets_common.sh"

failed=0

for preset in "${presets[@]}"; do
    name=$(preset_name "$preset")
    logfile=$(preset_logfile "$preset")
    lora_path=$(preset_lora_path "$preset")

    if [ ! -f "$lora_path" ]; then
        echo "SKIPPED diffusers smoke test for $name (no LoRA found at $lora_path, training did not produce one)" | tee -a "$logfile"
        echo "FAILED (diffusers load): $name" | tee -a "$ERROR_LOG"
        failed=1
        continue
    fi

    if ! venv/bin/python3 test/smoke_load_diffusers.py "$preset" "$lora_path" 2>&1 | tee -a "$logfile"; then
        echo "FAILED (diffusers load): $name" | tee -a "$ERROR_LOG"
        failed=1
        continue
    fi

    echo "OK (diffusers load): $name" | tee -a "$ERROR_LOG"
done

exit $failed
