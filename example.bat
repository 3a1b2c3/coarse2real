@echo off
setlocal enableextensions
cd /d "%~dp0"

REM Minimal smoke-test for C2R: 1 prompt x all shipped control videos, 320x576, 21 frames, 10 steps.
REM Output: outputs\c2r-example\
REM Expect a few minutes on a 30+ GB GPU. enable_vram_management=true for safer memory headroom.

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: .venv missing. Run setup.bat first.
    exit /b 2
)

call .venv\Scripts\python.exe -m inference.run_inference --config inference\config_example.json
exit /b %ERRORLEVEL%
