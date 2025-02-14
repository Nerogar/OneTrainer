@echo off
setlocal EnableDelayedExpansion

if not defined PYTHON (set "PYTHON=python")
if not defined VENV_DIR (set "VENV_DIR=%~dp0venv")

:check_python_version
for /f "delims=" %%v in ('%PYTHON% -c "import sys; print(sys.version)"') do set "PYTHON_VERSION=%%v"
%PYTHON% -c "import sys; sys.exit(0) if sys.version_info >= (3,10) and sys.version_info < (3,13) else sys.exit(1)"
if errorlevel 1 (
    echo OneTrainer requires Python 3.10.x, 3.11.x or 3.12.x.
    echo Your current Python version is: !PYTHON_VERSION!
    echo.
    echo Please install one of them from:
    echo https://www.python.org/downloads/windows/
    echo.
    echo Reminder: Do not rely on installation videos; they are often out of date or incorrect. Bypassing this message will lead to errors and support will not be provided.
    goto end_error
)


:check_venv
dir "%VENV_DIR%" >NUL 2>&1
if not errorlevel 1 goto activate_venv
echo Creating virtual environment in %VENV_DIR%
%PYTHON% -m venv "%VENV_DIR%"
if errorlevel 1 (
    echo Couldn't create venv.
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
echo Checking if CUDA is available...
for /f "tokens=*" %%i in ('CALL %PYTHON% -c "import torch; print(torch.cuda.is_available())"') do set "CUDA_AVAILABLE=%%i"
if "%CUDA_AVAILABLE%"=="True" goto end_success
set /p USE_ZLUDA=CUDA is not available. Are you using AMD GPUs on Windows? (y/n)
if /i "%USE_ZLUDA%"=="y" goto install_zluda
goto end_error

:install_zluda
echo Continuing with ZLUDA installation...
%PYTHON% scripts/install_zluda.py
goto end_success

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
