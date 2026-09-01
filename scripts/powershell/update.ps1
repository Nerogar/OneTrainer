$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Resolve-Path "$PSScriptRoot\..\.." ).Path

Write-Host "[OneTrainer] Updating OneTrainer to latest version from Git repository..."
& git pull
if ($LASTEXITCODE -ne 0) { throw "git pull failed with exit code $LASTEXITCODE." }

# reload ps library
. "$PSScriptRoot\lib.include.ps1"

Prepare-RuntimeEnvironment "upgrade"

Write-Host ""
Write-Host "[OneTrainer] Update completed successfully!"
Write-Host ""
