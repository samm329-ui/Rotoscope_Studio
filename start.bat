@echo off
REM Rotoscope Studio one-click starter for Windows.
REM This script does NOT install Python itself. It verifies
REM the environment, installs missing Python packages, then
REM launches the backend server.

setlocal
cd /d "%~dp0"

set "PYTHON_EXE=python"
where %PYTHON_EXE% >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not on PATH.
    echo Please install Python 3.9 or higher from https://www.python.org/
    pause
    exit /b 1
)

echo ============================================
echo Rotoscope Studio - one-click starter
echo ============================================
echo.

echo Step 1 of 3: checking environment ...
%PYTHON_EXE% scripts\setup_check.py
if errorlevel 1 (
    echo.
    echo Step 2 of 3: installing missing packages ...
    %PYTHON_EXE% scripts\install_deps.py
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies.
        pause
        exit /b 1
    )
) else (
    echo.
    echo Step 2 of 3: all packages already installed, skipping.
)

echo.
echo Step 3 of 3: starting backend server ...
echo Open http://127.0.0.1:8000 in your browser.
echo.

start "" http://127.0.0.1:8000
%PYTHON_EXE% -m uvicorn app.main:app --host 127.0.0.1 --port 8000

pause
endlocal