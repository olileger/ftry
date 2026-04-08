@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "REPO_ROOT=%%~fI"
set "ACTIVATE_BAT=%REPO_ROOT%\.venv\Scripts\activate.bat"

if not exist "%ACTIVATE_BAT%" (
    echo Virtual environment not found at "%ACTIVATE_BAT%".
    exit /b 1
)

call "%ACTIVATE_BAT%"
if errorlevel 1 exit /b %errorlevel%

python "%REPO_ROOT%\tests\run_all_tests.py" %*
exit /b %errorlevel%

