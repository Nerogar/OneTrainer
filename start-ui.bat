@echo off

if not defined PYTHON (set PYTHON=python)
if not defined VENV_DIR (set "VENV_DIR=%~dp0%venv")

:check_venv
dir %VENV_DIR% > NUL 2> NUL
if %ERRORLEVEL% == 0 goto :activate_venv
echo venv not found, please run install.bat first
goto :end

:activate_venv
echo activating venv %VENV_DIR%
set PYTHON="%VENV_DIR%\Scripts\python.exe"

:launch
%PYTHON% scripts\train_ui.py

:end
pause