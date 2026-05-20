@echo off
setlocal enableextensions enabledelayedexpansion
cd /d "%~dp0"

REM Full setup: Python 3.11 venv (uv) + PyTorch cu128 + flash-attn-4 + c2r package + model weights.
REM Uses `uv pip` because uv-created venvs don't bootstrap pip. Idempotent.

set "VENV=.venv"
set "PY=%~dp0%VENV%\Scripts\python.exe"
set "TORCH_INDEX=https://download.pytorch.org/whl/cu128"

REM Resolve uv: prefer PATH, fall back to %USERPROFILE%\.local\bin\uv.exe.
set UV_EXE=
where uv >nul 2>&1 && set UV_EXE=uv
if not defined UV_EXE if exist "%USERPROFILE%\.local\bin\uv.exe" set UV_EXE=%USERPROFILE%\.local\bin\uv.exe
if not defined UV_EXE (
    echo ERROR: uv not found on PATH and not at %USERPROFILE%\.local\bin\uv.exe.
    echo Install uv ^(https://docs.astral.sh/uv/^) or create the venv manually with Python 3.11 + pip.
    exit /b 2
)

echo ============================================================
echo [1/4] Create Python 3.11 venv at %VENV%
echo ============================================================
if exist "%PY%" (
    echo     SKIP - %PY% already exists
) else (
    call "%UV_EXE%" venv "%VENV%" --python 3.11
    if errorlevel 1 (
        echo ERROR - uv venv creation failed.
        exit /b 1
    )
)

echo.
echo ============================================================
echo [2/4] Install PyTorch 2.8.0 + torchvision + torchaudio ^(cu128^)
echo ============================================================
call "%UV_EXE%" pip install --python "%PY%" torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --extra-index-url %TORCH_INDEX%
if errorlevel 1 (
    echo ERROR - PyTorch install failed.
    exit /b 1
)

echo.
echo ============================================================
echo [3/4] Install c2r package ^(pulls requirements.txt^) + flash-attn-4
echo ============================================================
call "%UV_EXE%" pip install --python "%PY%" -e .
if errorlevel 1 (
    echo ERROR - c2r editable install failed.
    exit /b 1
)
echo Attempting flash-attn-4 ^(no Windows wheel published; SDPA fallback is fine if this fails^)...
call "%UV_EXE%" pip install --python "%PY%" flash-attn-4==4.0.0b10
if errorlevel 1 (
    echo WARNING - flash-attn-4 install failed. C2R will fall back to PyTorch SDPA attention backend.
)

echo.
echo ============================================================
echo [4/4] Download model weights
echo ============================================================
call "%~dp0download_models.bat"

echo.
echo === Setup complete ===
echo To run inference:
echo     %PY% -m inference.run_inference --config inference\config_1gpu.json
echo Or the minimal smoke test:
echo     example.bat
goto :eof
