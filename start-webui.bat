@echo off

REM Avoid footgun by explicitly navigating to the directory containing the batch file
cd /d "%~dp0"

REM Verify that OneTrainer is our current working directory
if not exist "scripts\web_ui.py" (
    echo Error: web_ui.py does not exist, please update your repository.
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

REM Disable HF_HUB_DISABLE_XET, buggy; default disables Xet (set to 0 to enable)
if not defined HF_HUB_DISABLE_XET (
    set "HF_HUB_DISABLE_XET=1"
)
echo HF_HUB_DISABLE_XET=%HF_HUB_DISABLE_XET%

:launch
echo Starting WebUI...
%PYTHON% scripts\web_ui.py
if errorlevel 1 (
    echo Error: WebUI script exited with code %ERRORLEVEL%
)

:end
pause
