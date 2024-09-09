@echo off

if not defined PYTHON (set PYTHON=python)
if not defined VENV_DIR (set "VENV_DIR=%~dp0%venv")

:check_venv
dir "%VENV_DIR%" > NUL 2> NUL
if %ERRORLEVEL% == 0 goto :activate_venv
echo venv not found, please run install.bat first
goto :end

:activate_venv
echo activating venv %VENV_DIR%
call "%VENV_DIR%\Scripts\activate.bat"
echo venv activated: %VENV_DIR%

set PYTHON=python
if defined PROFILE (set PYTHON=%PYTHON% -m scalene --off --cpu --gpu --profile-all --no-browser)
echo Using Python %PYTHON%

:launch
accelerate launch scripts\train_ui.py
if %ERRORLEVEL% NEQ 0 (
    echo Failed to launch with accelerate. Launching with regular Python.
    %PYTHON% scripts\train_ui.py
)

:end
pause