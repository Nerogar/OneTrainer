@echo off
setlocal EnableDelayedExpansion

if not defined GIT (
    set "GIT=git"
)
if not defined PYTHON (
    set "PYTHON=python"
)
if not defined VENV_DIR (
    set "VENV_DIR=%~dp0venv"
)

:git_pull
echo Checking repository and branch information...

REM Check if we're working with the official repo
FOR /F "tokens=* USEBACKQ" %%F IN (`"%GIT%" config --get remote.origin.url`) DO (
    set "remote_url=%%F"
)
echo Remote origin: %remote_url%

REM Get current branch name
FOR /F "tokens=* USEBACKQ" %%F IN (`"%GIT%" rev-parse --abbrev-ref HEAD`) DO (
    set "current_branch=%%F"
)
echo Current branch: %current_branch%

REM Compare current to expected repo and branch
set "is_official_repo="
echo %remote_url% | findstr /i "Nerogar/OneTrainer" >nul && set "is_official_repo=1"

if not defined is_official_repo (
    echo INFO: You are using a fork or custom repository.
    echo      This is normal if you've made your own modifications.
    echo      If unexpected, consider switching to the official Nerogar/OneTrainer repository.
)

if /I not "%current_branch%"=="master" (
    echo INFO: You are on branch %current_branch% instead of master.
    echo      This is normal if you're working on a specific branch.
    echo      If unexpected, switch to master with: git checkout master
)

echo Attempting to update current branch
"%GIT%" fetch origin
if errorlevel 1 (
    echo Error: Could not fetch updates
    goto :end_error
)

echo Pulling changes...
"%GIT%" pull
if errorlevel 1 (
    echo Error: Git pull failed.
    goto :end_error
)
goto :check_venv

:check_venv
dir "%VENV_DIR%" >NUL 2>NUL
if errorlevel 1 (
    echo Error: Virtual environment not found, please run install.bat first
    goto :end_error
) else (
    goto :check_python_version
)

:check_python_version
echo Checking Python version...
"%PYTHON%" --version
if errorlevel 1 (
    echo Error: Failed to get Python version
    goto :end_error
)

echo.
set "SUPPORTED_PY_VERSIONS=3.10.x, 3.11.x or 3.12.x"
"%PYTHON%" "%~dp0scripts\util\version_check.py" 3.10 3.13 2>&1
if errorlevel 1 (
    echo.
    goto :wrong_python_version
)
goto :activate_venv

:activate_venv
echo Activating virtual environment: %VENV_DIR%
set "PYTHON=%VENV_DIR%\Scripts\python.exe"

:install_dependencies
echo Installing dependencies...
echo Upgrading pip and setuptools...
"%PYTHON%" -m pip install --upgrade --upgrade-strategy eager pip setuptools
if errorlevel 1 (
    echo Error: pip upgrade failed.
    goto :end_error
)

echo Installing requirements (this may take a while)...
"%PYTHON%" -m pip install --upgrade --upgrade-strategy eager -r requirements.txt
if errorlevel 1 (
    echo Error: Installing requirements failed.
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
echo Please install Python %SUPPORTED_PY_VERSIONS% from:
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
exit /b %errorlevel%
