#Requires -Version 5.1
# Dot-source after lib.include.ps1. Only called for the rocm-windows environment.

function Update-RocmCheckout {
    $configPaths = @('pyproject.toml', 'pixi.lock', 'install.bat', 'start-ui.bat', 'update.bat')

    # A scoped stash can still record the index. Leave staged work and unfinished merges alone.
    $conflicts = @(& git diff --name-only --diff-filter=U)
    if ($LASTEXITCODE -ne 0) { throw 'Could not check Git conflicts.' }
    if ($conflicts.Count -gt 0) {
        throw 'Resolve the existing Git conflicts before updating.'
    }
    & git diff --cached --quiet
    if ($LASTEXITCODE -eq 1) {
        throw 'Commit or unstage your staged changes before using the AMD updater.'
    }
    if ($LASTEXITCODE -ne 0) { throw 'Could not check the Git index.' }

    $changed = @(& git diff --name-only -- @configPaths)
    if ($LASTEXITCODE -ne 0) { throw 'Could not check local AMD configuration changes.' }

    $backup = $null
    if ($changed.Count -gt 0) {
        $previousBackup = & git rev-parse --verify --quiet refs/stash
        if ($LASTEXITCODE -notin @(0, 1)) { throw 'Could not check existing Git stashes.' }
        $label = 'OneTrainer AMD update backup ' + [guid]::NewGuid().ToString('N')
        Invoke-Run git stash push -m $label -- @configPaths
        $backup = & git rev-parse --verify refs/stash
        if ($LASTEXITCODE -ne 0 -or -not $backup -or $backup -eq $previousBackup) {
            throw 'Could not identify the AMD configuration backup; update stopped.'
        }
        Write-OT "Local AMD configuration backed up in Git stash $backup."
    }

    $pullError = $null
    try {
        # Do not merge local commits or autostash unrelated edits through the user's Git config.
        Invoke-Run git pull --ff-only --no-autostash
    } catch {
        $pullError = $_
    }

    if ($backup) {
        # Reapply even after a failed pull, and retain the stash as a recovery copy.
        & git stash apply $backup
        $applyExit = $LASTEXITCODE
        $conflicts = @(& git diff --name-only --diff-filter=U)
        if ($LASTEXITCODE -ne 0) {
            throw "Could not check restored configuration. Backup retained in Git stash $backup."
        }

        $onlyLockConflict = $conflicts.Count -eq 1 -and $conflicts[0] -eq 'pixi.lock'
        if (-not $pullError -and ($applyExit -eq 0 -or $conflicts -contains 'pixi.lock')) {
            # Generated lockfiles use a binary merge driver. Start from upstream and let Pixi
            # regenerate the AMD entries after the manifest has been restored/merged.
            Invoke-Run git restore --source=HEAD --staged --worktree -- pixi.lock
        }

        if ($applyExit -ne 0 -and -not (-not $pullError -and $onlyLockConflict)) {
            throw ("Could not merge the local AMD configuration. Backup retained in Git stash $backup. " +
                'Resolve the conflicts, unstage the resolved files, then rerun install.bat with ' +
                'OT_PLATFORM=rocm-windows. No dependencies were installed.')
        }

        # A conflicted stash application stages its cleanly merged files. The preflight above
        # ensured these files were originally unstaged; preserve that state after recovery.
        Invoke-Run git restore --source=HEAD --staged -- @configPaths
        Write-OT "Local AMD configuration restored. Backup retained in Git stash $backup."
    }

    if ($pullError) { throw $pullError }
}
