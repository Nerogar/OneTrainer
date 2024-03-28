@echo off

if not defined PYTHON (set PYTHON=python)
if not defined VENV_DIR (set "VENV_DIR=%~dp0%venv")

:check_venv
dir "%VENV_DIR%" > NUL 2> NUL
if %ERRORLEVEL% == 0 goto :activate_venv
echo creating venv in %VENV_DIR%
%PYTHON% -m venv "%VENV_DIR%"
if %ERRORLEVEL% == 0 goto :activate_venv
echo Couldn't create venv
goto :end_error

:activate_venv
echo activating venv %VENV_DIR%
set PYTHON="%VENV_DIR%\Scripts\python.exe"

:install_dependencies
echo installing dependencies
%PYTHON% -m pip install -r requirements.txt

:check_cuda
echo checking if CUDA is available
for /f "tokens=*" %%i in ('%PYTHON% -c "import torch; print(torch.cuda.is_available())"') do set CUDA_AVAILABLE=%%i
if %CUDA_AVAILABLE% == "True" goto :end_success
set /p USE_ZLUDA=CUDA is not available. Are you using AMD GPUs on Windows? (y/n) 
if "%USE_ZLUDA%" == "y" goto :install_zluda
if "%USE_ZLUDA%" == "Y" goto :install_zluda
goto :end_error

:install_zluda
echo continue with ZLUDA
%PYTHON% scripts/install_zluda.py
goto :end_success

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