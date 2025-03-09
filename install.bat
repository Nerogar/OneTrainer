@echo off
setlocal EnableDelayedExpansion

REM Change to the directory containing the batch file to mitigate PEBCAK
cd /d "%~dp0"

if not defined PYTHON ( set "PYTHON=python" )
:: %~dp0 expands to the full directory path of the script (with trailing backslash)
if not defined VENV_DIR ( set "VENV_DIR=%~dp0venv" )

:: ----------------------------------------------------------------------------
:: 1. Check that a real Python version is available in PATH (ignoring the Windows Store alias)
:: ----------------------------------------------------------------------------
:check_python_exists
set "REAL_PYTHON="
for /f "delims=" %%p in ('where %PYTHON% 2^>nul ^| findstr /v /i "WindowsApps"') do (
    set "REAL_PYTHON=%%p"
    goto :found_valid_python
)


:found_valid_python
if "%REAL_PYTHON%"=="" (
    echo Error: Python was not found in your PATH. It is likely that you have not installed Python yet.
    goto :wrong_python_version
) else (
    set "PYTHON=%REAL_PYTHON%"
)

:: ----------------------------------------------------------------------------
:: 2. Check that Pythons version is supported by OT using version_check.py
:: ----------------------------------------------------------------------------

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

:: ----------------------------------------------------------------------------
:: 3. Continue with installation: create a virtual environment and install dependencies
:: ----------------------------------------------------------------------------
:check_venv
dir "%VENV_DIR%" >NUL 2>&1
if not errorlevel 1 goto :activate_venv
echo Creating virtual environment in %VENV_DIR%
"%PYTHON%" -m venv "%VENV_DIR%"
if errorlevel 1 (
    echo Error: Couldn't create virtual environment.
    echo Checking if venv module is installed...
    "%PYTHON%" -c "import venv" 2>nul
    if errorlevel 1 (
        echo Error: Python venv module not found. Please install it first:
        echo %PYTHON% -m pip install --user virtualenv
    )
    goto :end_error
)
goto :activate_venv

:activate_venv
echo Activating virtual environment: %VENV_DIR%
set "PYTHON=%VENV_DIR%\Scripts\python.exe"

:install_dependencies
echo Installing dependencies...
echo This may take a while, please be patient...
"%PYTHON%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo Error: Failed to install dependencies
    goto :end_error
)
echo Dependencies installed successfully

:check_cuda
echo Checking for CUDA support...
for /f "tokens=*" %%i in ('CALL "%PYTHON%" -c "import torch; print(torch.cuda.is_available())"') do set "CUDA_AVAILABLE=%%i"
if "%CUDA_AVAILABLE%"=="True" goto :end_success

echo CUDA is not available.
set "USE_ZLUDA="
:ask_zluda
set /p USE_ZLUDA=Are you using AMD GPUs on Windows? (y/n):
if /i "%USE_ZLUDA%"=="y" goto :install_zluda
if /i "%USE_ZLUDA%"=="n" goto :end_error
echo Invalid input. Please enter 'y' or 'n'.
goto :ask_zluda

:install_zluda
echo Continuing with ZLUDA installation...
"%PYTHON%" scripts\install_zluda.py
if errorlevel 1 (
    echo Error: ZLUDA installation failed
    goto :end_error
)
goto :end_success

:wrong_python_version
echo.
echo Please install a supported Python version from:
echo https://www.python.org/downloads/windows/
echo.
echo Reminder: Do not rely on installation videos; they are often out of date.
goto :end_error

:end_success
echo.
echo ************
echo Install done
echo ************
pause
exit /b 0

:end_error
echo.
echo ********************
echo Error during install
echo ********************
pause
exit /b 1
