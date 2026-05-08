@echo off
rem ─────────────────────────────────────────────────────────────
rem Bayes Tree — Windows installer
rem Creates a virtual environment and installs all dependencies.
rem ─────────────────────────────────────────────────────────────
setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
set "VENV_DIR=%SCRIPT_DIR%.venv"
set "REQ_FILE=%SCRIPT_DIR%requirements.txt"

echo.
echo   🌳 Bayes Tree — Installer
echo   ─────────────────────────────────────────────
echo.

rem ── Find Python ─────────────────────────────────────────────
set "PYTHON="

where python >nul 2>&1
if %errorlevel% equ 0 (
    for /f "delims=" %%v in ('python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"') do set "PY_VER=%%v"
    for /f "delims=" %%v in ('python -c "import sys; print(sys.version_info.major)"') do set "PY_MAJOR=%%v"
    for /f "delims=" %%v in ('python -c "import sys; print(sys.version_info.minor)"') do set "PY_MINOR=%%v"
    if !PY_MAJOR! geq 3 if !PY_MINOR! geq 8 (
        set "PYTHON=python"
    )
)

if not defined PYTHON (
    where python3 >nul 2>&1
    if %errorlevel% equ 0 (
        for /f "delims=" %%v in ('python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"') do set "PY_VER=%%v"
        for /f "delims=" %%v in ('python3 -c "import sys; print(sys.version_info.major)"') do set "PY_MAJOR=%%v"
        for /f "delims=" %%v in ('python3 -c "import sys; print(sys.version_info.minor)"') do set "PY_MINOR=%%v"
        if !PY_MAJOR! geq 3 if !PY_MINOR! geq 8 (
            set "PYTHON=python3"
        )
    )
)

if not defined PYTHON (
    echo   [ERROR] Python 3.8+ is required but not found.
    echo          Download from https://www.python.org/downloads/
    echo          Make sure to check "Add Python to PATH" during install.
    echo.
    pause
    exit /b 1
)

echo   [OK] Found Python !PY_VER! (!PYTHON!)

rem ── Create virtual environment ──────────────────────────────
if exist "%VENV_DIR%" (
    echo   [!] Virtual environment already exists at %VENV_DIR%
    set /p "answer=   Recreate it? [y/N] "
    if /i "!answer!"=="y" (
        echo   Removing old virtual environment...
        rmdir /s /q "%VENV_DIR%"
    ) else (
        echo   Keeping existing virtual environment
    )
)

if not exist "%VENV_DIR%" (
    echo   Creating virtual environment...
    !PYTHON! -m venv "%VENV_DIR%"
    if %errorlevel% neq 0 (
        echo   [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo   [OK] Virtual environment created
)

rem ── Install dependencies ────────────────────────────────────
echo   Installing dependencies...
"%VENV_DIR%\Scripts\pip.exe" install --upgrade pip --quiet
if %errorlevel% neq 0 (
    echo   [WARN] pip upgrade failed, continuing...
)

"%VENV_DIR%\Scripts\pip.exe" install -r "%REQ_FILE%" --quiet
if %errorlevel% neq 0 (
    echo   [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)
echo   [OK] Dependencies installed

rem ── Verify installation ─────────────────────────────────────
echo   Verifying installation...
"%VENV_DIR%\Scripts\python.exe" -c "import yaml, matplotlib, numpy; from reportlab.lib.pagesizes import A4; from PyQt6.QtWidgets import QApplication; print('All packages OK')" 2>nul
if %errorlevel% equ 0 (
    echo   [OK] All packages verified
) else (
    echo   [WARN] Some packages may not have loaded
)

rem ── Done ────────────────────────────────────────────────────
echo.
echo   ─────────────────────────────────────────────
echo   Installation complete!
echo.
echo   To run the CLI:
echo     %VENV_DIR%\Scripts\python.exe bayes-tree-eng.py examples\shroud.yaml
echo.
echo   To run the GUI:
echo     %VENV_DIR%\Scripts\python.exe bayes_tree_gui.py
echo.
echo   Or activate the environment first:
echo     %VENV_DIR%\Scripts\activate.bat
echo     python bayes_tree_gui.py
echo.
pause
