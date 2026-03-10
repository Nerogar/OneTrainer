. "$PSScriptRoot\lib.include.ps1"

Prepare-RuntimeEnvironment

Invoke-InEnv python "scripts/@args[1].py" @args[1..]
