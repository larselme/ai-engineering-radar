@echo off
setlocal
cd /d "%~dp0.."

if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment not found at %CD%\.venv
    exit /b 2
)

".venv\Scripts\python.exe" main.py
exit /b %ERRORLEVEL%