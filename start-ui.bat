@echo off

REM Change to the directory containing the batch file to mitigate PEBCAK
cd /d "%~dp0"

REM Check if the UI script exists before proceeding
if not exist "scripts\train_ui.py" (
    echo Error: UI script not found at scripts\train_ui.py
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
set PYTHON="%VENV_DIR%\Scripts\python.exe"
if defined PROFILE (set PYTHON=%PYTHON% -m scalene --off --cpu --gpu --profile-all --no-browser)
echo Using Python %PYTHON%

:launch
echo Starting UI...
%PYTHON% scripts\train_ui.py
if errorlevel 1 (
    echo Error: UI script exited with code %ERRORLEVEL%
)

:end
pause
