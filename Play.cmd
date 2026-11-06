@echo off
cd /d "%~dp0"
if not exist "runtime\ready.txt" (
    powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1"
    if errorlevel 1 goto failed
)
"runtime\python.exe" -B -X utf8 game\server.py
if not errorlevel 1 exit /b 0
:failed
pause
exit /b 1
