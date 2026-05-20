@echo off
setlocal enableextensions
cd /d "%~dp0"

REM Minimal smoke-test for C2R: 1 prompt x 1 control video, 320x576, 21 frames, 10 steps.
REM Output: outputs\c2r-example\
REM Expect a few minutes on a 30+ GB GPU. enable_vram_management=true for safer memory headroom.

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: .venv missing. Run setup.bat first.
    exit /b 2
)

REM Use the first shipped control video as the single example input.
if not exist "inference\example_control_videos" mkdir "inference\example_control_videos"
if not exist "inference\example_control_videos\coarse-control-video-1.mp4" (
    copy /Y "inference\control_videos\coarse-control-video-1.mp4" "inference\example_control_videos\coarse-control-video-1.mp4" >nul
)

call .venv\Scripts\python.exe -m inference.run_inference --config inference\config_example.json
exit /b %ERRORLEVEL%
