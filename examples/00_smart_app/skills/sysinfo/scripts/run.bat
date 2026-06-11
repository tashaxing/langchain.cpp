@echo off
REM run.bat -- System info collection script for Windows.

set FILTER=%~1
if "%FILTER%"=="" set FILTER=all

if "%FILTER%"=="cpu" (
    echo CPU: %NUMBER_OF_PROCESSORS% cores
    goto :eof
)

if "%FILTER%"=="memory" (
    systeminfo 2>nul | findstr /C:"Total Physical Memory" /C:"Available Physical Memory"
    if errorlevel 1 (
        echo Memory info unavailable
    )
    goto :eof
)

if "%FILTER%"=="disk" (
    wmic logicaldisk get size,freespace,caption 2>nul
    if errorlevel 1 (
        echo Disk info unavailable
    )
    goto :eof
)

if "%FILTER%"=="all" (
    echo === CPU ===
    echo CPU: %NUMBER_OF_PROCESSORS% cores
    echo.
    echo === Memory ===
    systeminfo 2>nul | findstr /C:"Total Physical Memory" /C:"Available Physical Memory"
    if errorlevel 1 (
        echo Memory info unavailable
    )
    echo.
    echo === Disk ===
    wmic logicaldisk get size,freespace,caption 2>nul
    if errorlevel 1 (
        echo Disk info unavailable
    )
    goto :eof
)

echo Unknown filter: %FILTER%. Use cpu, memory, disk, or all.
