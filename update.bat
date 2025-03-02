@echo off

if not defined GIT (
    set GIT=git
)
if not defined PYTHON (
    set PYTHON=python
)
if not defined VENV_DIR (
    set "VENV_DIR=%~dp0venv"
)

:git_pull
echo Attempting to update current branch
%GIT% fetch origin
if %ERRORLEVEL% NEQ 0 (
    echo Could not fetch updates
    goto :end_error
)

echo Pulling changes...
%GIT% pull
if %ERRORLEVEL% NEQ 0 (
    echo Git pull failed.
    goto :end_error
)

REM Warn if not on master branch; Assume user is competent to avoid breaking custom changes.
FOR /F "tokens=* USEBACKQ" %%F IN (`%GIT% rev-parse --abbrev-ref HEAD`) DO (
    set current_branch=%%F
)
if /I not "%current_branch%"=="master" (
    echo WARNING: You are on branch %current_branch%. To update master, please switch manually:
    echo         git checkout master
)

goto :check_venv

:check_venv
dir "%VENV_DIR%" > NUL 2> NUL
if %ERRORLEVEL% NEQ 0 (
    echo venv not found, please run install.bat first
    goto :end_error
) else (
    goto :check_python_version
)

:check_python_version
echo Checking Python version...
"%PYTHON%" --version
if %ERRORLEVEL% NEQ 0 (
    goto :end_error
)

echo.
set "CHECK_CMD="%PYTHON%" "%~dp0scripts\util\version_check.py" 3.10 3.13"
%CHECK_CMD% 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo.
    goto :wrong_python_version
)
goto :activate_venv

:activate_venv
echo activating venv %VENV_DIR%
set PYTHON="%VENV_DIR%\Scripts\python.exe"

:install_dependencies
echo installing dependencies
%PYTHON% -m pip install --upgrade --upgrade-strategy eager pip setuptools
if %ERRORLEVEL% NEQ 0 (
    echo pip upgrade failed.
    goto :end_error
)
%PYTHON% -m pip install --upgrade --upgrade-strategy eager -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo Installing requirements failed.
    goto :end_error
)

:end_success
echo.
echo ***********
echo Update done
echo ***********
goto :end

:wrong_python_version
echo.
echo Please install Python 3.10.x, 3.11.x or 3.12.x from:
echo https://www.python.org/downloads/windows/
echo.
echo Reminder: Do not rely on installation videos; they are often out of date.
goto :end_error

:end_error
echo.
echo *******************
echo Error during update
echo *******************
goto :end

:end
pause
