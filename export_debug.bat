@echo off

REM Change to the directory containing the batch file to mitigate PEBCAK
cd /d "%~dp0"

REM Check if the debug script exists before proceeding
if not exist "scripts\generate_debug_report.py" (
    echo Error: Debug script not found at scripts\generate_debug_report.py
    goto :end
)

if not defined PYTHON (set PYTHON=python)
if not defined VENV_DIR (set "VENV_DIR=%~dp0venv")

:check_venv
dir "%VENV_DIR%" >NUL 2>NUL
if not errorlevel 1 goto :activate_venv
echo venv not found, please run install.bat first
goto :end

:activate_venv
echo activating venv %VENV_DIR%
set PYTHON="%VENV_DIR%\Scripts\python.exe"
echo Using Python %PYTHON%

:launch
echo Generating debug report...
%PYTHON% scripts\generate_debug_report.py
if errorlevel 1 (
    echo Error: Debug report generation failed with code %ERRORLEVEL%
) else (
    echo Now upload the debug report to your Github issue or post in Discord.
)

:end
pause
