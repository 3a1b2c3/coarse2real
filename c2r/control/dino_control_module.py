import os
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


def _cached_hf_snapshot_from_disk(model_id: str) -> str | None:
    if "/" not in model_id:
        return None

    cache_roots: list[Path] = []
    hf_hub_cache = os.environ.get("HF_HUB_CACHE")
    transformers_cache = os.environ.get("TRANSFORMERS_CACHE")
    hf_home = os.environ.get("HF_HOME")
    if hf_hub_cache:
        cache_roots.append(Path(hf_hub_cache).expanduser())
    if transformers_cache:
        cache_roots.append(Path(transformers_cache).expanduser())
    if hf_home:
        cache_roots.append(Path(hf_home).expanduser() / "hub")
    cache_roots.append(Path.home() / ".cache" / "huggingface" / "hub")

    repo_cache_name = "models--" + model_id.replace("/", "--")
    seen_roots: set[Path] = set()
    for root in cache_roots:
        if root in seen_roots:
            continue
        seen_roots.add(root)
        snapshots_dir = root / repo_cache_name / "snapshots"
        if not snapshots_dir.is_dir():
            continue
        snapshots = sorted(
            (path for path in snapshots_dir.iterdir() if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for snapshot in snapshots:
            if (snapshot / "config.json").is_file():
                return str(snapshot)
    return None


def _resolve_dino_model_source(model_id: str, local_files_only: bool) -> tuple[str, bool]:
    local_path = Path(model_id).expanduser()
    if local_path.exists():
        return str(local_path), True

    try:
        from huggingface_hub import snapshot_download

        snapshot = snapshot_download(repo_id=model_id, local_files_only=True)
        return snapshot, True
    except Exception:
        snapshot = _cached_hf_snapshot_from_disk(model_id)
        if snapshot is not None:
            return snapshot, True

    return model_id, local_files_only


class DINOFeaturesExtractor:
    """Inference-only DINO features extractor for control video frames."""

    def __init__(
        self,
        model_id: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        frame_height: int = 480,
        frame_width: int = 832,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        local_files_only: bool = False,
    ):
        self.device = torch.device(device)
        self.dtype = dtype
        self.frame_height = frame_height
        self.frame_width = frame_width
        model_source, resolved_local_files_only = _resolve_dino_model_source(model_id, local_files_only)
        self.model_source = model_source
        self.model = AutoModel.from_pretrained(
            model_source,
            dtype=dtype,
            local_files_only=resolved_local_files_only,
        ).eval().to(self.device, dtype=self.dtype)
        self.processor = AutoImageProcessor.from_pretrained(
            model_source,
            local_files_only=resolved_local_files_only,
        )
        patch = getattr(self.model.config, "patch_size", 16)
        self.patch_size = patch
        self.Hp = frame_height // patch
        self.Wp = frame_width // patch

    def to(self, device=None, dtype=None, non_blocking=False):
        if device is not None:
            self.device = torch.device(device)
        if dtype is not None:
            self.dtype = dtype
        self.model.to(device=self.device, dtype=self.dtype, non_blocking=non_blocking)
        return self

    def set_resolution(self, frame_height: int, frame_width: int):
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.Hp = frame_height // self.patch_size
        self.Wp = frame_width // self.patch_size
        return self

    @staticmethod
    def _to_pil_frames(frames: Union[List[Image.Image], torch.Tensor]) -> List[Image.Image]:
        if isinstance(frames, torch.Tensor):
            tensor = frames
            if tensor.dtype == torch.uint8:
                tensor = tensor.float().div(255.0)
            pil_frames = []
            for index in range(tensor.shape[0]):
                image = tensor[index]
                if image.ndim != 3:
                    raise ValueError(f"Expected frame tensor [H, W, C], got shape {image.shape}")
                image = image.detach().cpu().numpy()
                if image.max() <= 1.0:
                    image = (image * 255.0).clip(0, 255)
                image = image.astype(np.uint8)
                pil_frames.append(Image.fromarray(image, mode="RGB"))
            return pil_frames
        return list(frames)

    @torch.no_grad()
    def get_features(self, frames: Union[List[Image.Image], torch.Tensor]):
        pil_frames = self._to_pil_frames(frames)
        inputs = self.processor(
            images=pil_frames,
            return_tensors="pt",
            do_center_crop=False,
            size={"height": self.frame_height, "width": self.frame_width},
        )
        pixel_values = inputs["pixel_values"].to(self.device, dtype=self.dtype)
        hidden_states = self.model(pixel_values).last_hidden_state  # [T, tokens, D]

        patch_tokens_count = self.Hp * self.Wp
        register_tokens_count = hidden_states.shape[1] - 1 - patch_tokens_count
        if register_tokens_count < 0:
            raise ValueError(
                f"Unexpected token layout from DINO model. Got {hidden_states.shape[1]} tokens, "
                f"but expected at least {1 + patch_tokens_count}."
            )

        patch_tokens = hidden_states[:, 1 + register_tokens_count :, :]
        patch_tokens = patch_tokens.unflatten(1, (self.Hp, self.Wp))  # [T, Hp, Wp, D]
        return patch_tokens


class TemporalDownsample(nn.Module):
    def __init__(
        self,
        c_in: int = 768,
        t_in: int = 81,
        t_out: int = 21,
        stride: int | None = None,
        kernel: int | None = None,
    ):
        super().__init__()
        if stride is None:
            stride = max(1, round(t_in / t_out))
        if kernel is None:
            kernel = 2 * stride + 1
        pad_t = kernel // 2
        self.conv = nn.Conv3d(
            in_channels=c_in,
            out_channels=c_in,
            kernel_size=(kernel, 1, 1),
            stride=(stride, 1, 1),
            padding=(pad_t, 0, 0),
            groups=c_in,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "b t h w c -> b c t h w")
        x = self.conv(x)
        x = rearrange(x, "b c t h w -> b t h w c")
        return x


class DINO2WanLatentAdapter(nn.Module):
    def __init__(self, Cd=768, C=5120, Td=81, T=21):
        super().__init__()
        self.temporal_downsample = TemporalDownsample(Cd, Td, T)
        self.ln = nn.LayerNorm(Cd)
        self.proj = nn.Linear(Cd, C)
        self.gate = nn.Parameter(torch.tensor(0.01))

    def forward(self, dino_patches):
        if dino_patches.ndim != 4:
            raise ValueError(f"Expected DINO patches [T, H, W, C], got {dino_patches.shape}")
        x = dino_patches.unsqueeze(0)  # [1, T, H, W, C]
        x = self.temporal_downsample(x)
        x = self.ln(x)
        x = self.proj(x)
        x = self.gate * x
        x = rearrange(x, "b t h w c -> b (t h w) c").contiguous()
        return x
