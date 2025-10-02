@echo off

REM Avoid footgun by explictly navigating to the directory containing the batch file
cd /d "%~dp0"

REM Verify that OneTrainer is our current working directory
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
if defined PROFILE (set PYTHON=%PYTHON% -m scalene --off --cpu --gpu --profile-all --no-browser)
echo Using Python %PYTHON%

REM Disable HF_HUB_DISABLE_XET, buggy; default disables Xet (set to 0 to enable) - https://github.com/Nerogar/OneTrainer/issues/949
if not defined HF_HUB_DISABLE_XET (
    set "HF_HUB_DISABLE_XET=1"
)
echo HF_HUB_DISABLE_XET=%HF_HUB_DISABLE_XET%
echo.
echo NOTE: Xet disabled, to enable it set as 0 before launch

:check_python_version
echo Checking Python version...
%PYTHON% --version
if errorlevel 1 (
    echo Error: Failed to get Python version
    goto :end_error
)

echo.
%PYTHON% "%~dp0scripts\util\version_check.py" 3.10 3.13 2>&1
if errorlevel 1 (
    echo.
    goto :wrong_python_version
)

:launch
echo Starting UI...
%PYTHON% scripts\train_ui.py
if errorlevel 1 (
    echo Error: UI script exited with code %ERRORLEVEL%
)

:end
pause
