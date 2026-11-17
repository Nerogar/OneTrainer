$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Resolve-Path "$PSScriptRoot\..\.." ).Path

. "$PSScriptRoot\lib.include.ps1"

Write-Host "[OneTrainer] Updating OneTrainer to latest version from Git repository..."
if ((Get-Platform) -eq 'rocm-windows') {
    # Keep ROCM selection after git pulling
    $env:OT_PLATFORM = 'rocm-windows'
    . "$PSScriptRoot\update-rocm.ps1"
    Update-RocmCheckout
} else {
    Invoke-Run git pull
}

# reload ps library
$script:_OT_LIB_INCLUDED = $false
. "$PSScriptRoot\lib.include.ps1"

Prepare-RuntimeEnvironment "upgrade"

Write-Host ""
Write-Host "[OneTrainer] Update completed successfully!"
Write-Host ""
