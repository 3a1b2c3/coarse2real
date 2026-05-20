@echo off
setlocal enableextensions
cd /d "%~dp0"

REM Thin wrapper around download_models.py — uses the project .venv if present,
REM else whatever python is on PATH (must have huggingface_hub installed).

if exist ".venv\Scripts\python.exe" (
    set "PY=.venv\Scripts\python.exe"
) else (
    set "PY=python"
)

call "%PY%" download_models.py %*
exit /b %ERRORLEVEL%
