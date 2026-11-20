@echo off

if "%OT_PIP_INSTALL%" == "true" (
    echo "[OneTrainer] Updating dependencies via pip..."
    CALL "%~dp0scripts\pip-install\update.bat"
    exit /b
)

chcp 65001 >nul
cd /d "%~dp0"
powershell -ExecutionPolicy Bypass -File "%~dp0scripts\powershell\update.ps1" %*
set "_EXIT=%ERRORLEVEL%"
pause
exit /b %_EXIT%
