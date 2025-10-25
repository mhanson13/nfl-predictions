@echo off
setlocal enabledelayedexpansion
set START_YEAR=2002
for /f %%Y in ('powershell -NoProfile -Command "(Get-Date).Year"') do set CURRENT_YEAR=%%Y
set YEARS=
for /L %%Y in (%START_YEAR%,1,%CURRENT_YEAR%) do (
    if defined YEARS (
        set YEARS=!YEARS! %%Y
    ) else (
        set YEARS=%%Y
    )
)
echo YEARS=!YEARS!
