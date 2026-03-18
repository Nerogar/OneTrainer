@echo off
chcp 65001 >nul
cd /d "%~dp0"
powershell -ExecutionPolicy Bypass -File "%~dp0scripts\powershell\update.ps1" %*
set "_EXIT=%ERRORLEVEL%"
pause
exit /b %_EXIT%
