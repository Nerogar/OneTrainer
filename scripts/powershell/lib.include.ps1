#Requires -Version 5.1

# USAGE: Dot-source this file from your script:
#   . "$PSScriptRoot\lib.include.ps1"

# guard against including the library multiple times
if ($script:_OT_LIB_INCLUDED) { return }
$script:_OT_LIB_INCLUDED = $true

$ErrorActionPreference = 'Stop'

# detect and set absolute path to the project root
$script:SCRIPT_DIR = (Resolve-Path "$PSScriptRoot\..\.." ).Path
Set-Location -LiteralPath $script:SCRIPT_DIR

# IMPORTANT: Don't modify the code below! Pass these variables via the environment!
if (-not $env:OT_CUDA_LOWMEM_MODE) { $env:OT_CUDA_LOWMEM_MODE = "false" }
if (-not $env:OT_PLATFORM)         { $env:OT_PLATFORM = "detect" }
if (-not $env:OT_SCRIPT_DEBUG)     { $env:OT_SCRIPT_DEBUG = "false" }

# Try this if encountering CUDA out-of-memory situations. YMMV
if ($env:OT_CUDA_LOWMEM_MODE -eq "true") {
    $env:PYTORCH_ALLOC_CONF = "backend:native,garbage_collection_threshold:0.6,max_split_size_mb:128,expandable_segments:True"
}


#region Logging

function Write-OT {
    param([Parameter(Position = 0)][string]$Message)
    Write-Host "[OneTrainer] $Message"
}

function Write-OTWarning {
    param([Parameter(Position = 0)][string]$Message)
    Write-Host "[OneTrainer] Warning: $Message" -ForegroundColor Yellow
}

function Write-OTError {
    param([Parameter(Position = 0)][string]$Message)
    Write-Host "[OneTrainer] Error: $Message" -ForegroundColor Red
}

function Write-OTDebug {
    param([Parameter(Position = 0)][string]$Message)
    if ($env:OT_SCRIPT_DEBUG -eq "true") {
        Write-OT "Debug: $Message"
    }
}

function Write-OTCommand {
    # Displays the command being executed for logging purposes.
    $parts = foreach ($a in $args) {
        if ("$a" -match '\s') { "`"$a`"" } else { "$a" }
    }
    Write-Host "[OneTrainer] + $($parts -join ' ')"
}

#endregion

#region Utility

function Test-CommandExists {
    param([Parameter(Position = 0)][string]$Command)
    $null -ne (Get-Command $Command -ErrorAction SilentlyContinue)
}

function Invoke-Run {
    # Exec and log if error
    Write-OTCommand @args
    $cmd = $args[0]
    if ($args.Length -gt 1) {
        $cmdArgs = $args[1..($args.Length - 1)]
        & $cmd @cmdArgs
    } else {
        & $cmd
    }
    if ($LASTEXITCODE -and $LASTEXITCODE -ne 0) {
        throw "Command '$cmd' failed with exit code $LASTEXITCODE."
    }
}

#endregion

#region Platform Detection

function Get-Platform {
    $platform = $env:OT_PLATFORM
    if ($platform -ne "detect") {
        Write-OTDebug "Using user-specified platform: $platform"
        return $platform
    }

    # check if nvidia
    if (Test-CommandExists "nvidia-smi") {
        Write-OTDebug "NVIDIA GPU detected via nvidia-smi."
        return "cuda"
    }

    # fallback to cpu, rocm not supported officially.
    Write-OTDebug "No supported CUDA GPU detected, falling back to CPU. ROCM is not officially supported on Windows by OneTrainer."
    return "cpu"
}

#endregion

#region Pixi

function Get-OrUpdatePixi {
    if (Test-CommandExists "pixi") {
        Write-OTDebug "'pixi' found, updating."
        Invoke-Run pixi self-update
    } else {
        Write-OTDebug "'pixi' not found, attempting installation."
        Write-OT "Installing pixi package manager..."

        # Install pixi using the official install script.
        iex (irm -useb https://pixi.sh/install.ps1)

        # current session may not reflect pixi in path, add manually to be sure
        $pixiBinDir = Join-Path $env:USERPROFILE ".pixi\bin"
        if ((Test-Path $pixiBinDir) -and ($env:Path -notlike "*$pixiBinDir*")) {
            $env:Path = "$pixiBinDir;$env:Path"
        }

        if (-not (Test-CommandExists "pixi")) {
            throw "Failed to install pixi. Please mention to the OneTrainer team. You can also try to install it manually: https://pixi.sh"
        }
        Write-OT "pixi installed successfully."
    }
}

function Install-Env {
    Invoke-Run pixi install --locked -e $script:OT_PLATFORM
}

function Invoke-InEnv {
    Invoke-Run pixi run --locked -e $script:OT_PLATFORM @args
}

#endregion

function Prepare-RuntimeEnvironment {
    Get-OrUpdatePixi

    # Detect the GPU platform and child processes use it.
    $script:OT_PLATFORM = Get-Platform
    $env:OT_PLATFORM = $script:OT_PLATFORM
    Write-OT "Platform: $($script:OT_PLATFORM)"

    Install-Env
}
