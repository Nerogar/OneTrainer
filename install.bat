@echo off

if not defined PYTHON (set PYTHON=python)
if not defined VENV_DIR (set "VENV_DIR=%~dp0%venv")

:check_venv
dir %VENV_DIR% > NUL 2> NUL
if %ERRORLEVEL% == 0 goto :activate_venv
echo creating venv in %VENV_DIR%
%PYTHON% -m venv %VENV_DIR%
if %ERRORLEVEL% == 0 goto :activate_venv
echo Couldn't create venv
goto :end_error

:activate_venv
echo activating venv %VENV_DIR%
set PYTHON="%VENV_DIR%\Scripts\python.exe"

:install_dependencies
echo installing dependencies
%PYTHON% -m pip install -r requirements.txt

:end_success
echo.
echo ************
echo Install done
echo ************
goto:end

:end_error
echo.
echo ********************
echo Error during install
echo ********************
goto:end

:end
pause