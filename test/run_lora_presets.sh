#!/bin/bash
set -o pipefail

source "$(dirname "$0")/lora_presets_common.sh"

> "$ERROR_LOG"

failed=0

for preset in "${presets[@]}"; do
    name=$(preset_name "$preset")
    logfile=$(preset_logfile "$preset")
    lora_path=$(preset_lora_path "$preset")
    echo "=== $name ===" | tee "$logfile"

    if ! ./run-cmd.sh train --preset-path "$preset" --config-path "$CONFIG" --config-value "output_model_destination=$lora_path" 2>&1 | tee -a "$logfile" || grep -q "Error during sampling" "$logfile"; then
        echo "FAILED (train): $name" | tee -a "$ERROR_LOG"
        failed=1
    else
        echo "OK (train): $name" | tee -a "$ERROR_LOG"
    fi
done

exit $failed
