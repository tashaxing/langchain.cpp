@echo off
REM stop.bat -- Stop the 00_smart_app server on Windows.
REM This script lives in bin\; the app base is the parent directory.

setlocal

set "PROCESS_NAME=00_smart_app.exe"

tasklist /FI "IMAGENAME eq %PROCESS_NAME%" 2>NUL | findstr /I /C:"%PROCESS_NAME%" >NUL
if %ERRORLEVEL% neq 0 (
    echo 00_smart_app is not running
    exit /b 0
)

echo Stopping 00_smart_app...
taskkill /IM "%PROCESS_NAME%" >NUL 2>&1

"%SystemRoot%\System32\timeout.exe" /t 2 /nobreak >NUL
tasklist /FI "IMAGENAME eq %PROCESS_NAME%" 2>NUL | findstr /I /C:"%PROCESS_NAME%" >NUL
if %ERRORLEVEL% == 0 (
    echo Force killing...
    taskkill /F /IM "%PROCESS_NAME%" >NUL 2>&1
)

echo Stopped.
