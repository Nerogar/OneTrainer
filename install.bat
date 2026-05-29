@echo off

if "%OT_PIP_INSTALL%" == "true" (
    echo "[OneTrainer] Installing dependencies via pip..."
    CALL scripts/pip-install/install.bat
    exit /b %_EXIT%
)

chcp 65001 >nul
cd /d "%~dp0"
powershell -ExecutionPolicy Bypass -File "%~dp0scripts\powershell\install.ps1" %*
set "_EXIT=%ERRORLEVEL%"
pause
exit /b %_EXIT%
