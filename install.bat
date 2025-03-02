@echo off
setlocal EnableDelayedExpansion

if not defined PYTHON (set "PYTHON=python")
:: %~dp0 expands to the full directory path of the script (with trailing backslash)
if not defined VENV_DIR (set "VENV_DIR=%~dp0venv")

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
::    (Currently supported versions: 3.10.x, 3.11.x, or 3.12.x)
:: ----------------------------------------------------------------------------
:check_python_version
echo Checking Python version...
"%PYTHON%" --version
if errorlevel 1 (
    goto end_error
)

echo.
set "CHECK_CMD="%PYTHON%" "%~dp0scripts\util\version_check.py" 3.10 3.13"
%CHECK_CMD% 2>&1
if errorlevel 1 (
    echo.
    goto :wrong_python_version
)

:: ----------------------------------------------------------------------------
:: 3. Continue with installation: create a virtual environment and install dependencies
:: ----------------------------------------------------------------------------
:check_venv
dir "%VENV_DIR%" >NUL 2>&1
if not errorlevel 1 goto activate_venv
echo Creating virtual environment in %VENV_DIR%
%PYTHON% -m venv "%VENV_DIR%"
if errorlevel 1 (
    echo Couldn't create virtual environment.
    goto end_error
)
goto activate_venv

:activate_venv
echo Activating virtual environment: %VENV_DIR%
set "PYTHON=%VENV_DIR%\Scripts\python.exe"

:install_dependencies
echo Installing dependencies...
%PYTHON% -m pip install -r requirements.txt
if errorlevel 1 goto end_error

:check_cuda
for /f "tokens=*" %%i in ('CALL %PYTHON% -c "import torch; print(torch.cuda.is_available())"') do set "CUDA_AVAILABLE=%%i"
if "%CUDA_AVAILABLE%"=="True" goto end_success
set /p USE_ZLUDA=CUDA is not available. Are you using AMD GPUs on Windows? (y/n)
if /i "%USE_ZLUDA%"=="y" goto install_zluda
goto end_error

:install_zluda
echo Continuing with ZLUDA installation...
%PYTHON% scripts\install_zluda.py
goto end_success

:wrong_python_version
echo.
echo Please install Python 3.10.x, 3.11.x or 3.12.x from:
echo https://www.python.org/downloads/windows/
echo.
echo Reminder: Do not rely on installation videos; they are often out of date.
goto end_error

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
