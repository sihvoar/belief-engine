@echo off
rem Bayes Tree launcher — auto-installs and runs the GUI or CLI
rem Usage:
rem   bayes-tree.bat                          launch GUI
rem   bayes-tree.bat examples\shroud.yaml     GUI with file
rem   bayes-tree.bat --cli examples\shroud.yaml  CLI mode
setlocal

set "DIR=%~dp0.."
set "VENV=%DIR%\.venv"
set "PY=%VENV%\Scripts\python.exe"

if not exist "%VENV%" (
    echo First run — installing dependencies...
    call "%DIR%\scripts\install.bat"
    echo.
)

if "%~1"=="--cli" (
    shift
    "%PY%" "%DIR%\scripts\bayes-tree-eng.py" %1 %2 %3 %4 %5 %6 %7 %8 %9
) else (
    start "" "%PY%" "%DIR%\scripts\bayes_tree_gui.py" %*
)
