@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion

rem --- Color codes ---
set "RED=[31m" & set "YEL=[33m" & set "GRN=[92m" & set "CYAN=[36m" & set "RESET=[0m"

rem --- Constants ---
pushd "%~dp0" || call :die "Cannot cd to script directory"
set "SCRIPT_DIR=%CD%"
set "VENV_DIR=%SCRIPT_DIR%\venv"
set "VERSION_FILE=%SCRIPT_DIR%\scripts\util\version_check.py"
set "MIN_PY=3.10" & set "MAX_PY=3.13"

goto :main

rem --- Helpers ---
:die
  echo.
  echo %RED%ERROR:%RESET% %~1&
  echo
  pause
  popd
  exit /b 1

:warn_store
  echo.
  echo %YEL% ?? WARNING: Windows Store Python detected ?? %RESET%
  echo Windows Store Python has a known history of causing insidious issues with virtual environments due to how
  echo Microsoft sandboxes it.
  echo.
  echo We strongly recommend installing Python directly from[36m https://www.python.org[0m instead.
  echo.
  echo Support for Windows Store Python is provided AS IS.
  set /p "ans=Proceed anyway? (y/n): "   >nul
  if /i "!ans!"=="y" exit /b 0
  exit /b 1

:wrong_python_version
    echo.
    echo Please install a supported Python version from:
    echo https://www.python.org/downloads/windows/
    echo.
    echo Reminder: Do not rely on installation videos; they are often out of date.
    exit /b 1

:run_or_die
  echo %~1
  cmd /c "%~1" || call :die "%~2"
  exit /b 0

rem --- Main ---
:main
rem 1) Check Python Launcher for available versions
for /f "tokens=*" %%V in ('py --list ^| findstr /c:"-V:"') do (
    rem parse version
    py -%%V "%VERSION_FILE%" %MIN_PY% %MAX_PY% >nul 2>&1 && (
    set "PYTHON=py -%%V"
    goto :py_ok
    )
)

rem 2) Try non-Store python next
for /f "delims=" %%P in ('where python 2^>nul ^| findstr /v /i "WindowsApps"') do (
  %%P "%VERSION_FILE%" %MIN_PY% %MAX_PY% >nul 2>&1 && (
    set "PYTHON=%%P"
    goto :py_ok
  )
)

rem 3) Finally as a failsafe try to ask about Windows Store python
for /f "delims=" %%P in ('where python 2^>nul ^| findstr /i "WindowsApps"') do (
  call :warn_store
  %%P "%VERSION_FILE%" %MIN_PY% %MAX_PY% >nul 2>&1 || (
    call :wrong_python_version
    call :die "Unsupported Python version %MIN_PY%-%MAX_PY% required"
  )
  set "PYTHON=%%P"
  goto :py_ok
)

rem If we reach here, no Python was found at all
echo %RED%ERROR: No Python installation found%RESET%
call :wrong_python_version
exit /b 1

:py_ok
echo %GRN%Using: %PYTHON% %RESET%

rem 4) Create & activate venv
if not exist "%VENV_DIR%\Scripts\python.exe" (
  echo Creating venv ...
  %PYTHON% -m venv "%VENV_DIR%" || call :die "venv creation failed"
)
set "PYTHON=%VENV_DIR%\Scripts\python.exe"

rem 5) Upgrade pip & install
call :run_or_die "%PYTHON% -m pip install --upgrade pip" "pip upgrade failed"
call :run_or_die "%PYTHON% -m pip install -r requirements.txt" "Dependencies install failed"

rem 6) Check CUDA
%PYTHON% -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"
if errorlevel 1 (
  set /p "ans=CUDA not found. AMD GPU? (y/n): " >nul
  if /i "!ans!"=="y" (
    call :run_or_die "%PYTHON% scripts\install_zluda.py" "ZLUDA install failed"
  ) else (
    call :die "CUDA unavailable - aborting"
  )
)

echo.
echo %GRN%**** Install successful! ****%RESET%
echo.
pause
popd
exit /b 0
