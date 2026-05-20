@echo off
setlocal enableextensions
cd /d "%~dp0"

REM Full setup: Python 3.11 venv + PyTorch cu128 + flash-attn-4 + c2r package + model weights.
REM Idempotent: skips steps whose outputs already exist. Continues past flash-attn failure on Windows.

set "VENV=.venv"
set "PY=%VENV%\Scripts\python.exe"
set "TORCH_INDEX=https://download.pytorch.org/whl/cu128"

echo ============================================================
echo [1/5] Create Python 3.11 venv at %VENV%
echo ============================================================
if exist "%PY%" (
    echo     SKIP - %PY% already exists
) else (
    where uv >nul 2>&1
    if errorlevel 1 (
        echo ERROR - uv not found on PATH. Install uv ^(https://docs.astral.sh/uv/^) or create the venv manually with Python 3.11.
        exit /b 1
    )
    call uv venv "%VENV%" --python 3.11
    if errorlevel 1 (
        echo ERROR - uv venv creation failed.
        exit /b 1
    )
)

echo.
echo ============================================================
echo [2/5] Upgrade pip
echo ============================================================
call "%PY%" -m pip install --upgrade pip

echo.
echo ============================================================
echo [3/5] Install PyTorch 2.8.0 + torchvision + torchaudio ^(cu128^)
echo ============================================================
call "%PY%" -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --extra-index-url %TORCH_INDEX%
if errorlevel 1 (
    echo ERROR - PyTorch install failed.
    exit /b 1
)

echo.
echo ============================================================
echo [4/5] Install c2r package ^(pulls requirements.txt^) + flash-attn-4
echo ============================================================
call "%PY%" -m pip install -e .
if errorlevel 1 (
    echo ERROR - c2r editable install failed.
    exit /b 1
)
echo Attempting flash-attn-4 ^(no Windows wheel published; SDPA fallback is fine if this fails^)...
call "%PY%" -m pip install flash-attn-4==4.0.0b10
if errorlevel 1 (
    echo WARNING - flash-attn-4 install failed. C2R will fall back to PyTorch SDPA attention backend.
)

echo.
echo ============================================================
echo [5/5] Download model weights
echo ============================================================
call "%~dp0download_models.bat"

echo.
echo === Setup complete ===
echo To run inference:
echo     %PY% -m inference.run_inference --config inference\config_1gpu.json
goto :eof
