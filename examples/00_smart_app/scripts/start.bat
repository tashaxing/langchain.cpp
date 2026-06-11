@echo off
REM start.bat -- Start the 00_smart_app server on Windows.
REM This script lives in bin\; the app base is the parent directory.

setlocal enabledelayedexpansion

set "PROCESS_NAME=00_smart_app.exe"
set "APP_DIR=%~dp0.."
set "BINARY=%APP_DIR%\bin\%PROCESS_NAME%"
set "CONFIG=%APP_DIR%\config\app_config.xml"
set "LOGDIR=%APP_DIR%\log"

if defined SMART_APP_DIR (
    set "APP_DIR=%SMART_APP_DIR%"
    set "BINARY=%APP_DIR%\bin\%PROCESS_NAME%"
    set "CONFIG=%APP_DIR%\config\app_config.xml"
    set "LOGDIR=%APP_DIR%\log"
)

if not exist "%BINARY%" (
    echo Error: binary not found: %BINARY%
    echo Please build and install the project first.
    exit /b 1
)

if not exist "%LOGDIR%" mkdir "%LOGDIR%"

tasklist /FI "IMAGENAME eq %PROCESS_NAME%" 2>NUL | findstr /I /C:"%PROCESS_NAME%" >NUL
if %ERRORLEVEL% == 0 (
    echo 00_smart_app is already running
    exit /b 0
)

echo Starting 00_smart_app...
start /B "" "%BINARY%" --config "%CONFIG%" > "%APP_DIR%\smart_app.out" 2>&1

"%SystemRoot%\System32\timeout.exe" /t 1 /nobreak >NUL
tasklist /FI "IMAGENAME eq %PROCESS_NAME%" 2>NUL | findstr /I /C:"%PROCESS_NAME%" >NUL
if %ERRORLEVEL% neq 0 (
    echo Error: 00_smart_app exited immediately.
    exit /b 1
)

echo Started.
echo Config: %CONFIG%
echo Logs:   %LOGDIR%\
