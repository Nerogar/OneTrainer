@echo off

if not defined GIT (set GIT=git)
if not defined PYTHON (set PYTHON=python)
if not defined VENV_DIR (set "VENV_DIR=%~dp0%venv")

:git_pull
echo pulling updates
%GIT% pull
if %ERRORLEVEL% == 0 goto :check_venv
echo could not pull updates
goto :end_error

:check_venv
dir "%VENV_DIR%" > NUL 2> NUL
if %ERRORLEVEL% == 0 goto :activate_venv
echo venv not found, please run install.bat first
goto :end_error

:activate_venv
echo activating venv %VENV_DIR%
set PYTHON="%VENV_DIR%\Scripts\python.exe"

:install_dependencies
echo installing dependencies
%PYTHON% -m pip install -r requirements-global.txt -r requirements-cuda.txt --force-reinstall

:end_success
echo.
echo ***********
echo Update done
echo ***********
goto:end

:end_error
echo.
echo *******************
echo Error during update
echo *******************
goto:end

:end
pause