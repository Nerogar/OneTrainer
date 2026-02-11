@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion

rem --- Color codes ---
set "RED=[31m" & set "YEL=[33m" & set "GRN=[92m" & set "CYAN=[36m" & set "RESET=[0m"

rem --- Constants ---
pushd "%~dp0" || call :die "Cannot cd to script directory"
set "SCRIPT_DIR=%CD%"
set "VENV_DIR=%SCRIPT_DIR%\\venv"
set "VERSION_FILE=%SCRIPT_DIR%\\scripts\\util\\version_check.py"
set "MIN_PY=3.10" & set "MAX_PY=3.13"

goto :main

rem --- Helpers ---
:die
  echo.
  echo %RED%ERROR:%RESET% %~1
  echo.
  pause
  popd
  (echo %CMDCMDLINE% | find /I "%~nx0" >nul) && exit /b 1 || exit 1

:warn_store
  echo.
  echo %YEL% WARNING: Possible Windows Store Python detected %RESET%
  echo Windows Store Python has a known history of causing insidious issues with virtual environments due to how
  echo Microsoft sandboxes it.
  echo.
  echo We strongly recommend installing Python directly from %CYAN%https://www.python.org%RESET% instead.
  echo.
  echo Support for Windows Store Python is provided AS IS.
  set "ans="
  set /p "ans=Proceed anyway? (y/n): "
  if /i "!ans!"=="y" exit /b 0
  exit /b 1

:wrong_python_version_message
    echo.
    echo %RED%No suitable Python version found or selected.%RESET%
    echo Please install a supported Python version ^(%MIN_PY% - ^< %MAX_PY%^) from:
    echo %CYAN%https://www.python.org/downloads/windows/%RESET%
    echo.
    echo Reminder: Do not rely on installation videos; they are often out of date.
    exit /b 1

:run_or_die
  echo Executing: %~1
  cmd /c "%~1" || call :die "Command failed: %~2"
  exit /b 0

rem --- Main ---
:main
echo %CYAN%Searching for a suitable Python installation...%RESET%
set "PYTHON="

if not exist "%VERSION_FILE%" (
    call :die """%VERSION_FILE%\" not found"
    goto :final_python_failure_handling
)

rem --- Python Detection ---
echo %CYAN%Step 1: Checking for Python in PATH (to support Conda installs)...%RESET%
where python >nul 2>&1
if not errorlevel 1 (
    for /f "delims=" %%P in ('where python') do (
        if not defined PYTHON (
            echo   Testing Python from PATH: "%%P"
            "%%P" "%VERSION_FILE%" %MIN_PY% %MAX_PY% >nul 2>&1
            if not errorlevel 1 (
                rem Check if the path is NOT under System32 to avoid the store stub
                echo "%%P" | findstr /I /V /C:"%SystemRoot%\System32" >nul
                if not errorlevel 1 (
                    echo   %GRN%SELECTED Python from PATH: "%%P"%RESET%
                    set "PYTHON=%%P"
                ) else (
                    echo   %YEL%Skipping system-level Python stub: "%%P"%RESET%
                )
            ) else (
                echo   %YEL%Python from PATH during step one is not a suitable version.%RESET%
            )
        )
    )
) else (
    echo %YEL%No 'python' found in PATH. Proceeding with other checks.%RESET%
)

rem If we found a valid PATH python version, we can skip the rest of the checks
if defined PYTHON goto :py_ok


echo.
rem 2) Loop through ver reported by py
echo %CYAN%Step 2: Scanning Python installations reported by "py --list"...%RESET%

if defined PYTHON goto :py_ok

set "PYTHON_VERSION_FROM_PY_LIST="
for /f "tokens=2 delims=:" %%L in ('py --list 2^>nul ^| findstr /R /C:"-V:[0-9][.][0-9]"') do (
    for /f "tokens=1" %%V in ("%%L") do (
        set "CURRENT_PY_VER_TO_TEST=%%V"
        echo   Testing Python !CURRENT_PY_VER_TO_TEST! via py.exe ...
        py -!CURRENT_PY_VER_TO_TEST! "%VERSION_FILE%" %MIN_PY% %MAX_PY% >nul 2>&1
        if not errorlevel 1 (
            echo   %GRN%SELECTED Python !CURRENT_PY_VER_TO_TEST! via py.exe%RESET%
            set "PYTHON=py -!CURRENT_PY_VER_TO_TEST!"
            set "PYTHON_VERSION_FROM_PY_LIST=!CURRENT_PY_VER_TO_TEST!"
            goto :found_python_via_py_list
        ) else (
            echo   %YEL%Python !CURRENT_PY_VER_TO_TEST! via py.exe is not suitable or version_check.py failed.%RESET%
        )
    )
    rem Check if we found Python and need to exit outer loop
    if defined PYTHON_VERSION_FROM_PY_LIST goto :found_python_via_py_list
)

:found_python_via_py_list
if not defined PYTHON_VERSION_FROM_PY_LIST (
    echo %YEL%No suitable Python version found via "py --list" that satisfies %MIN_PY% ^>= v ^< %MAX_PY%.%RESET%
    rem PYTHON remains unset, script will proceed to the next step
)

rem Check if PYTHON was set by found_python_via_py_list logic. If so, go to :py_ok.
if defined PYTHON goto :py_ok

rem 3) If py launcher fails, search common install directories
echo.
echo %CYAN%Step 3: Searching for Python in common installation directories...%RESET%
set "SEARCH_PATHS="%ProgramFiles%\Python" "%LOCALAPPDATA%\Programs\Python""
for %%D in (%SEARCH_PATHS%) do (
    if exist "%%~D" (
        for /d %%P in ("%%~D\Python*") do (
            if exist "%%P\python.exe" (
                if not defined PYTHON (
                    echo   Testing "%%P\python.exe"...
                    "%%P\python.exe" "%VERSION_FILE%" %MIN_PY% %MAX_PY% >nul 2>&1
                    if not errorlevel 1 (
                        echo   %GRN%SELECTED Python from "%%P"%RESET%
                        set "PYTHON=%%~P\python.exe"
                        goto :py_ok
                    ) else (
                        echo   %YEL%"%%P\python.exe" is not a suitable version.%RESET%
                    )
                )
            )
        )
    )
)
echo %YEL%No suitable Python found in common directories.%RESET%

rem 4) Finally as a failsafe try to ask about Windows Store python, only if not found yet
if not defined PYTHON (
    echo.
    echo %CYAN%Step 4: Checking for Windows Store Python installations in PATH...%RESET%
    set "STORE_PYTHON_CHECKED="
    for /f "delims=" %%P in ('where python 2^>nul ^| findstr /i "WindowsApps"') do (
      if defined PYTHON ( goto :py_ok_check_step4 )
      set "STORE_PYTHON_CHECKED=true"
      echo Found potential Windows Store Python at "%%P".
      call :warn_store
      if errorlevel 1 (
        echo %YEL%  ^> Skipping Store Python "%%P" due to user choice or warning issue.%RESET%
      ) else (
        REM User agreed to use this Store Python (warn_store returned 0)
        echo Testing agreed-upon Store Python at "%%P"...
        "%%P" "%VERSION_FILE%" %MIN_PY% %MAX_PY%
        set "LAST_ERRORLEVEL=!errorlevel!"
        echo   ^> Exit code from '"%%P" "%VERSION_FILE%"': !LAST_ERRORLEVEL!

        if !LAST_ERRORLEVEL! == 0 (
          echo %GRN%  ^> Using selected Store Python: "%%P"%RESET%
          set "PYTHON=%%P"
          goto :py_ok_check_step4
        ) else (
          echo %RED%  ^> Version check failed for this agreed-upon Store Python "%%P" ^(Code: !LAST_ERRORLEVEL!^).%RESET%
          call :wrong_python_version_message
          REM After calling wrong_python_version_message, which tries to exit, we must ensure this path also exits.
          echo %RED%ERROR: The selected Windows Store Python version is not supported.%RESET%
          pause
          popd
          exit /b 1
        )
      )
    )
    :py_ok_check_step4
    if defined PYTHON ( goto :py_ok )

    if not defined STORE_PYTHON_CHECKED (
        echo %YEL%No Windows Store Python installations found in PATH during step 4.%RESET%
    )
)

rem If we reach here and PYTHON is not set, no suitable Python was found at all
if not defined PYTHON (
  echo.
  call :wrong_python_version_message
  REM The above call prints details and sets errorlevel. Now, exit the main script.
  echo.
  echo %RED%ERROR: Failed to find a supported Python version after all checks.%RESET%
  echo Please ensure a Python version between %MIN_PY% and %MAX_PY% is available.
  pause
  popd
  exit /b 1
)

:final_python_failure_handling
exit /b 0

:py_ok
if not defined PYTHON (
    echo %RED%Internal error: Reached :py_ok without PYTHON being set. This should not happen.%RESET%
    call :die "Script logic error at :py_ok."
)
echo.
echo %GRN%Using Python: !PYTHON!%RESET%

rem 4) Create & activate venv
echo.
echo %CYAN%Managing virtual environment...%RESET%
if not exist "%VENV_DIR%\\Scripts\\python.exe" (
  echo Creating venv at "%VENV_DIR%"...
  "!PYTHON!" -m venv "%VENV_DIR%" || call :die "venv creation failed using !PYTHON!"
) else (
  echo Virtual environment already exists at "%VENV_DIR%"
)
set "PYTHON_VENV=%VENV_DIR%\\Scripts\\python.exe"
if not exist "%PYTHON_VENV%" (
    call :die "Virtual environment Python executable not found at '%PYTHON_VENV%' after venv creation/check."
)
set "PYTHON=%PYTHON_VENV%"

echo Activating virtual environment...
call "%VENV_DIR%\Scripts\activate.bat"
echo Virtual environment activated.

rem  Check for Tkinter
echo %CYAN%Checking for Tkinter availability...%RESET%
python -c "import tkinter,sys; sys.exit(0 if hasattr(tkinter,'TkVersion') else 1)" >nul 2>&1
if not errorlevel 1 goto :tk_ok

echo %RED%Tkinter not found%RESET%
call :die "Re-run the Python installer and enable 'tcl/tk and IDLE' (its enabled by default on fresh installations, re-enable/dont turn it off)"
goto :EOF

:tk_ok
echo %GRN%Tkinter is available, proceeding ... %RESET%

rem 5) Upgrade pip & install
echo.
echo %CYAN%Upgrading pip and installing dependencies from requirements.txt...%RESET%
echo Executing: python -m pip install --upgrade pip
python -m pip install --upgrade pip || call :die "pip upgrade failed"
echo Executing: python -m pip install -r requirements.txt
python -m pip install -r requirements.txt || call :die "Dependencies install failed"

rem 6) Check CUDA
echo.
echo %CYAN%Checking CUDA availability...%RESET%
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"
if errorlevel 1 (
  echo %YEL%CUDA not found via torch.cuda.is_available.%RESET%
  set "ans_amd="
  set /p "ans_amd=AMD GPU? (y/n): "
  if /i "!ans_amd!"=="y" (
    echo Executing: python "%SCRIPT_DIR%\scripts\install_zluda.py"
    python "%SCRIPT_DIR%\scripts\install_zluda.py" || call :die "ZLUDA install failed"
  ) else (
    call :die "CUDA unavailable and not an AMD GPU setup - aborting. Please check PyTorch and NVIDIA driver compatibility."
  )
) else (
    echo %GRN%CUDA is available.%RESET%
)

echo.
echo %GRN%**** Install successful^^! ****%RESET%
echo.
pause
popd
exit /b 0
