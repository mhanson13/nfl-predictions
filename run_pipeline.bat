@echo off
setlocal EnableExtensions EnableDelayedExpansion

python -m tools.run_pipeline %*
set EXITCODE=%ERRORLEVEL%

endlocal & exit /b %EXITCODE%

