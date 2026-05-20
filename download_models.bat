@echo off
setlocal enableextensions
cd /d "%~dp0"

set "WAN_MARKER=models\wan\Wan2.1_VAE.pth"
set "DIT_MARKER=models\pretrained_dit_backbone\c2r-dit-backbone-14B.safetensors"
set "ADAPTER_MARKER=models\dino_adapter\c2r-dino-adapter.safetensors"
set "DINO_MARKER=models\dino\dinov3-vitb16-pretrain-lvd1689m\config.json"

echo === Wan2.1-T2V-14B base ===
if exist "%WAN_MARKER%" (echo     SKIP ^(already present^)) else (if not exist "models\wan" mkdir "models\wan" & call hf download Wan-AI/Wan2.1-T2V-14B --local-dir models\wan)

echo === C2R DiT backbone (gated) ===
if exist "%DIT_MARKER%" (echo     SKIP ^(already present^)) else (if not exist "models\pretrained_dit_backbone" mkdir "models\pretrained_dit_backbone" & call hf download gonsaBRK/coarse2real c2r-dit-backbone-14B.safetensors --local-dir models\pretrained_dit_backbone)

echo === C2R DINO adapter (gated) ===
if exist "%ADAPTER_MARKER%" (echo     SKIP ^(already present^)) else (if not exist "models\dino_adapter" mkdir "models\dino_adapter" & call hf download gonsaBRK/coarse2real c2r-dino-adapter.safetensors --local-dir models\dino_adapter)

echo === DINOv3 ViT-B/16 ===
if exist "%DINO_MARKER%" (echo     SKIP ^(already present^)) else (if not exist "models\dino\dinov3-vitb16-pretrain-lvd1689m" mkdir "models\dino\dinov3-vitb16-pretrain-lvd1689m" & call hf download facebook/dinov3-vitb16-pretrain-lvd1689m --local-dir models\dino\dinov3-vitb16-pretrain-lvd1689m)

echo.
echo === Summary ===
for %%D in ("models\wan" "models\pretrained_dit_backbone" "models\dino_adapter" "models\dino\dinov3-vitb16-pretrain-lvd1689m") do call :report %%~D
echo.
echo If a gated download failed, request access at https://huggingface.co/gonsaBRK/coarse2real then re-run.
echo Set "dino_model_path" in inference\config_1gpu.json to "models/dino/dinov3-vitb16-pretrain-lvd1689m" after DINOv3 is present.
goto :eof

:report
if exist "%~1" (powershell -NoProfile -Command "$b=(Get-ChildItem -Recurse -Force '%~1' -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum; '{0,14:N0} bytes ({1,6:N2} GB)  %~1' -f $b, ($b/1GB)") else (echo     (missing^)            %~1)
goto :eof
