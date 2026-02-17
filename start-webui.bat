@echo off

REM Avoid footgun by explicitly navigating to the directory containing the batch file
cd /d "%~dp0"

REM Verify that OneTrainer is our current working directory
if not exist "scripts\train_webui.py" (
    echo Error: train_webui.py does not exist, you have done something very wrong. Reclone the repository.
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
if not defined WEBUI_HOST (set "WEBUI_HOST=127.0.0.1")
if not defined WEBUI_PORT (set "WEBUI_PORT=7860")

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

echo.
echo Starting WebUI on http://%WEBUI_HOST%:%WEBUI_PORT%
%PYTHON% scripts\train_webui.py --host %WEBUI_HOST% --port %WEBUI_PORT%
if errorlevel 1 (
    echo Error: WebUI script exited with code %ERRORLEVEL%
)

:end
pause
