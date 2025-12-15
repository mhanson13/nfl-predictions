@REM Copyright (c) 2025 Matt Hanson
@REM
@REM Licensed under the Apache License, Version 2.0 (the "License");
@REM you may not use this file except in compliance with the License.
@REM You may obtain a copy of the License at
@REM
@REM     http://www.apache.org/licenses/LICENSE-2.0
@REM
@REM Unless required by applicable law or agreed to in writing, software
@REM distributed under the License is distributed on an "AS IS" BASIS,
@REM WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
@REM See the License for the specific language governing permissions and
@REM limitations under the License.

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
