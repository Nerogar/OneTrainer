. "$PSScriptRoot\lib.include.ps1"

Prepare-RuntimeEnvironment

Invoke-InEnv python "scripts/train_ui.py" @args
