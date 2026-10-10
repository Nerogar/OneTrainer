@echo off

if "%OT_PIP_INSTALL%" == "true" (
    echo "[OneTrainer] Running UI inside virtual environment..."
    CALL "%~dp0scripts\pip-install\start-ui.bat"
    exit /b
)

chcp 65001 >nul
cd /d "%~dp0"
powershell -ExecutionPolicy Bypass -File "%~dp0scripts\powershell\start-ui.ps1" %*
set "_EXIT=%ERRORLEVEL%"
pause
exit /b %_EXIT%
