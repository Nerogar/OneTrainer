. "$PSScriptRoot\lib.include.ps1"

Prepare-RuntimeEnvironment

Invoke-InEnv python "scripts/generate_debug_report.py"

Write-Host ""
Write-OT "Please upload the above debug report to your GitHub issue or post in Discord."
