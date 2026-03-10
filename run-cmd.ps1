# Xet is buggy. Disabled by default unless already defined - https://github.com/Nerogar/OneTrainer/issues/949
if (-not $env:HF_HUB_DISABLE_XET) { $env:HF_HUB_DISABLE_XET = "1" }

. "$PSScriptRoot\scripts\powershell\lib.include.ps1"

if ($args.Length -eq 0) {
    Write-OTError 'You must provide the name of the script to execute, such as "train".'
    exit 1
}

$OT_CUSTOM_SCRIPT_FILE = "scripts/$($args[0]).py"
if (-not (Test-Path $OT_CUSTOM_SCRIPT_FILE)) {
    Write-OTError "Custom script file `"$OT_CUSTOM_SCRIPT_FILE`" does not exist."
    exit 1
}

Prepare-RuntimeEnvironment

$remainingArgs = if ($args.Length -gt 1) { $args[1..($args.Length - 1)] } else { @() }
Invoke-InEnv python $OT_CUSTOM_SCRIPT_FILE @remainingArgs
