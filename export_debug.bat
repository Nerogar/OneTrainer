@echo off

if "%OT_PIP_INSTALL%" == "true" (
    echo "[OneTrainer] Running debug export inside virtual environment..."
    CALL "%~dp0scripts\pip-install\export_debug.bat"
    exit /b
)

chcp 65001 >nul
cd /d "%~dp0"
powershell -ExecutionPolicy Bypass -File "%~dp0scripts\powershell\export_debug.ps1" %*
set "_EXIT=%ERRORLEVEL%"
pause
exit /b %_EXIT%
