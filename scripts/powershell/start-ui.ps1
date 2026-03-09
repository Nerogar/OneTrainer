# Xet is buggy. Disabled by default unless already defined. #TODO remove on Xet update
# https://github.com/Nerogar/OneTrainer/issues/949
if (-not $env:HF_HUB_DISABLE_XET) {
    $env:HF_HUB_DISABLE_XET = "1"
}

. "$PSScriptRoot\lib.include.ps1"

Prepare-RuntimeEnvironment

Invoke-InEnv python "scripts/train_ui.py" @args
