@echo off
setlocal EnableDelayedExpansion

REM Avoid footgun by explictly navigating to the directory containing the batch file
cd /d "%~dp0"

REM Verify that OneTrainer is our current working directory
if not exist "scripts\train_ui.py" (
    echo Error: train_ui.py does not exist, you have done something very wrong. Reclone the repository.
    goto :end
)

if not defined GIT ( set "GIT=git" )
if not defined PYTHON ( set "PYTHON=python" )
if not defined VENV_DIR ( set "VENV_DIR=%~dp0venv" )

:git_pull
echo Checking repository and branch information...

REM Get current branch name
FOR /F "tokens=* USEBACKQ" %%F IN (`"%GIT%" rev-parse --abbrev-ref HEAD`) DO (
    set "current_branch=%%F"
)
echo Current branch: %current_branch%

REM Determine tracking information (remote and branch)
set "tracking_info="
FOR /F "tokens=* USEBACKQ" %%F IN (`"%GIT%" rev-parse --abbrev-ref --symbolic-full-name @{upstream} 2^>NUL`) DO (
    set "tracking_info=%%F"
)

if not defined tracking_info (
    echo INFO: Current branch has no tracking remote configured.
    echo      This is normal for local-only branches.
    echo      Updates cannot be pulled automatically. Configure tracking with:
    echo      git branch --set-upstream-to=origin/master %current_branch%
) else (
    for /F "tokens=1,2 delims=/" %%a in ("!tracking_info!") do (
        set "tracking_remote=%%a"
        set "tracking_branch=%%b"
    )
    echo Tracking: !tracking_info!

    FOR /F "tokens=* USEBACKQ" %%F IN (`"!GIT!" config --get remote.!tracking_remote!.url 2^>NUL`) DO (
        set "remote_url=%%F"
    )
    echo Remote !tracking_remote!: !remote_url!

    set "is_official_repo="
    echo !remote_url! | findstr /i "Nerogar/OneTrainer" >nul && set "is_official_repo=1"

    set "is_master_branch="
    if /I "!tracking_branch!"=="master" (set "is_master_branch=1")

    if not defined is_official_repo (set "non_standard_setup=1")
    if not defined is_master_branch (set "non_standard_setup=1")

    if defined non_standard_setup (
        echo INFO: Non-standard repository setup detected:
        if not defined is_official_repo echo        - Using non-official repository: !remote_url!
        if not defined is_master_branch echo        - On branch !tracking_branch! instead of master
        echo      This is normal if you're using a fork or working on a specific branch.
    )

    REM Get current commit hash
    FOR /F "tokens=* USEBACKQ" %%F IN (`"%GIT%" rev-parse HEAD`) DO (
        set "local_commit=%%F"
    )
    echo Local commit: !local_commit:~0,8!...

    echo Fetching updates...
    "%GIT%" fetch !tracking_remote!
    if errorlevel 1 (
        echo Error: Could not fetch updates
        goto :end_error
    )

    REM Get remote commit hash
    FOR /F "tokens=* USEBACKQ" %%F IN (`"%GIT%" rev-parse !tracking_remote!/!tracking_branch!`) DO (
        set "remote_commit=%%F"
    )
    echo Remote commit: !remote_commit:~0,8!...

    if "!local_commit!"=="!remote_commit!" (
        echo Repository is already up to date, skipping pull.
    ) else (
        echo Updates available, pulling changes...
        "%GIT%" pull
        if errorlevel 1 (
            echo Error: Git pull failed.
            goto :end_error
        )
    )
)

goto :check_venv

:check_venv
dir "%VENV_DIR%" >NUL 2>NUL
if errorlevel 1 (
    echo Error: Virtual environment not found, please run install.bat first
    goto :end_error
) else (
    goto :activate_venv
)

:activate_venv
echo Activating virtual environment: %VENV_DIR%
set "PYTHON=%VENV_DIR%\Scripts\python.exe"
goto :check_python_version

:check_python_version
echo Checking Python version...
"%PYTHON%" --version
if errorlevel 1 (
    echo Error: Failed to get Python version
    goto :end_error
)

echo.
"%PYTHON%" "%~dp0scripts\util\version_check.py" 3.10 3.13 2>&1
if errorlevel 1 (
    echo.
    goto :wrong_python_version
)
goto :install_dependencies

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
echo Please install a supported Python version from:
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
