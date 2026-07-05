@echo off
cd /d "%~dp0"

set "OT_DEV="
for %%a in (%*) do (
    if "%%a"=="--dev" set "OT_DEV=1"
)

if not exist "scripts\train_ui.py" (
    echo Error: train_ui.py does not exist, you have done something very wrong. Reclone the repository.
    goto :end
)

if not defined PYTHON (
    where python >NUL 2>NUL
    if errorlevel 1 (
        echo Error: Python is not installed or not in PATH
        goto :end
    )
    set PYTHON=python
)
if not defined VENV_DIR (set "VENV_DIR=%~dp0venv")

:check_venv
dir "%VENV_DIR%" > NUL 2> NUL
if not errorlevel 1 goto :activate_venv
echo venv not found, please run install.bat first
goto :end

:activate_venv
echo activating venv %VENV_DIR%
if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo Error: Python executable not found in virtual environment
    goto :end
)
set PYTHON="%VENV_DIR%\Scripts\python.exe" -X utf8
echo Using Python %PYTHON%

REM Disable Xet (buggy) - https://github.com/Nerogar/OneTrainer/issues/949
if not defined HF_HUB_DISABLE_XET (
    set "HF_HUB_DISABLE_XET=1"
)

:check_python_version
%PYTHON% --version
if errorlevel 1 (
    echo Error: Failed to get Python version
    goto :end
)
%PYTHON% "%~dp0scripts\util\version_check.py" 3.10 3.14 2>&1
if errorlevel 1 (
    goto :end
)

:check_node
where node >NUL 2>NUL
if errorlevel 1 (
    echo Error: Node.js is not installed or not in PATH
    echo Please install Node.js from https://nodejs.org/
    goto :end
)

:check_gui_built
if not exist "web\gui\dist\main\main\index.cjs" (
    echo Error: Web GUI has not been built yet.
    echo Please run install.bat or update.bat first to build the web UI.
    goto :end
)

:launch
echo Starting OneTrainer Web UI...
cd web\gui
call npx electron .
cd ..\..
if errorlevel 1 (
    echo Error: Web UI exited with code %ERRORLEVEL%
)

:end
pause
