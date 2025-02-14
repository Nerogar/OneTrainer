@echo off

if not defined GIT (set GIT=git)
if not defined PYTHON (set PYTHON=python)
if not defined VENV_DIR (set "VENV_DIR=%~dp0%venv")

:git_pull
echo Attempting to update current branch
%GIT% fetch origin
if %ERRORLEVEL% NEQ 0 (
    echo Could not fetch updates
    goto :end_error
)

%GIT% pull
if %ERRORLEVEL% == 0 goto :check_venv

echo Current branch pull failed, switching to master
FOR /F "tokens=* USEBACKQ" %%F IN (`%GIT% rev-parse --abbrev-ref HEAD`) DO SET current_branch=%%F
%GIT% checkout -f master
if %ERRORLEVEL% NEQ 0 (
    echo Could not switch to master branch
    goto :end_error
)

%GIT% pull origin master
if %ERRORLEVEL% NEQ 0 (
    echo Could not pull updates
    goto :end_error
)
goto :check_venv

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
%PYTHON% -m pip install --upgrade --upgrade-strategy eager pip setuptools
%PYTHON% -m pip install --upgrade --upgrade-strategy eager -r requirements.txt

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
