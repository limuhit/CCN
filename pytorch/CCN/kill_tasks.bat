echo off

for /f "tokens=2 delims= " %%c in ('tasklist /M CCN.cp37-win_amd64.pyd') do (
    SET "var="&for /f "delims=0123456789" %%i in ("%%c") do set var=%%i
    if defined var (echo %%c NOT numeric) else (
    echo %%c
    taskkill /F /PID %%c
    )
)