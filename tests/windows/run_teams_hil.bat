@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..\..") do set "REPO_ROOT=%%~fI"
set "ACTIVATE_BAT=%REPO_ROOT%\.venv\Scripts\activate.bat"

if not exist "%ACTIVATE_BAT%" (
    echo Virtual environment not found at "%ACTIVATE_BAT%".
    exit /b 1
)

call "%ACTIVATE_BAT%"
if errorlevel 1 exit /b %errorlevel%

call ftry pop -t "%REPO_ROOT%\samples\teams\hil-grp-feature-debate-team\team.yaml" -p "Should we add live match alerts to our football fan mobile app this quarter?"
if errorlevel 1 exit /b %errorlevel%

call ftry pop -t "%REPO_ROOT%\samples\teams\hil-han-support-routing-team\team.yaml" -p "I was charged twice for my football premium subscription and I need help."
if errorlevel 1 exit /b %errorlevel%

call ftry pop -t "%REPO_ROOT%\samples\teams\hil-mag-launch-planning-team\team.yaml" -p "We are launching a weekly football digest for coaches next month. Build a lightweight launch brief."
if errorlevel 1 exit /b %errorlevel%

call ftry pop -t "%REPO_ROOT%\samples\teams\hil-seq-support-brief-team\team.yaml" -p "Customer says the football match report export failed twice and wants a clear status update today."
exit /b %errorlevel%
