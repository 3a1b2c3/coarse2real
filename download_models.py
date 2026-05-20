"""
Downloads all C2R model weights into the *standard* HuggingFace cache
(~/.cache/huggingface/hub/models--<org>--<name>/) and patches
inference/config_1gpu.json with the resolved snapshot paths.

Per-repo behaviour:
  - Wan-AI/Wan2.1-T2V-14B          full repo
  - gonsaBRK/coarse2real (gated)   two files (DiT backbone + DINO adapter)
  - facebook/dinov3-vitb16-...     full repo

Idempotent: snapshot_download() resumes partial downloads. Continues past
per-repo failures (gated access, network) and reports a summary.

Run from the coarse2real repo root:
    .\.venv\Scripts\python.exe download_models.py
"""

import json
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

REPO_ROOT = Path(__file__).resolve().parent
CONFIG_PATHS = [
    REPO_ROOT / "inference" / "config_1gpu.json",
    REPO_ROOT / "inference" / "config_example.json",
    REPO_ROOT / "inference" / "config_multigpu_usp.json",
    REPO_ROOT / "inference" / "config_multigpu_dp.json",
]

JOBS = [
    {
        "repo_id": "Wan-AI/Wan2.1-T2V-14B",
        "allow_patterns": None,
        "config_key": "base_model_dir",
    },
    {
        # umt5-xxl tokenizer files; c2r's resolve_tokenizer_config walks
        # base_model_dir/google/umt5-xxl. Pull into the local models/wan dir so
        # mixed setups (manually staged text encoder + VAE) still resolve.
        "repo_id": "Wan-AI/Wan2.1-T2V-14B",
        "allow_patterns": ["google/umt5-xxl/*"],
        "local_dir": str(REPO_ROOT / "models" / "wan"),
        "label": "tokenizer",
    },
    {
        "repo_id": "gonsaBRK/coarse2real",
        "allow_patterns": ["c2r-dit-backbone-14B.safetensors"],
        "config_key": "dit_path",
        "config_join": "c2r-dit-backbone-14B.safetensors",
    },
    {
        "repo_id": "gonsaBRK/coarse2real",
        "allow_patterns": ["c2r-dino-adapter.safetensors"],
        "config_key": "dino_adapter_path",
        "config_join": "c2r-dino-adapter.safetensors",
    },
    {
        "repo_id": "facebook/dinov3-vitb16-pretrain-lvd1689m",
        "allow_patterns": None,
        "config_key": "dino_model_path",
    },
]


def download_one(job: dict) -> Path | None:
    repo_id = job["repo_id"]
    label_suffix = ""
    if job.get("label"):
        label_suffix = f"   ({job['label']})"
    elif job.get("allow_patterns"):
        label_suffix = f"   ({job['allow_patterns'][0]})"
    print(f"=== {repo_id}{label_suffix} ===")
    kwargs = {
        "repo_id": repo_id,
        "allow_patterns": job.get("allow_patterns"),
        # max_workers=1 serializes the per-file downloads to avoid the
        # tqdm+concurrent.futures TimeoutError seen on Python 3.12.
        "max_workers": 1,
    }
    if job.get("local_dir"):
        kwargs["local_dir"] = job["local_dir"]
    try:
        path = snapshot_download(**kwargs)
        print(f"    OK -> {path}")
        return Path(path)
    except GatedRepoError as e:
        print(f"    GATED — request access at https://huggingface.co/{repo_id} ({e})")
    except RepositoryNotFoundError as e:
        print(f"    NOT FOUND — {e}")
    except Exception as e:
        print(f"    FAILED ({type(e).__name__}): {e}")
    return None


def patch_config(updates: dict) -> None:
    for config_path in CONFIG_PATHS:
        if not config_path.exists():
            print(f"\nConfig not found at {config_path}; skipping.")
            continue
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        changed = []
        for key, value in updates.items():
            if key not in cfg:
                continue
            old = cfg.get(key)
            new = str(value).replace("\\", "/")
            if old != new:
                cfg[key] = new
                changed.append((key, old, new))
        if not changed:
            print(f"\n{config_path.name}: already up to date.")
            continue
        config_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
        print(f"\nPatched {config_path}:")
        for key, old, new in changed:
            print(f"  {key}: {old!r} -> {new!r}")


def main() -> int:
    updates: dict = {}
    failed = []
    for job in JOBS:
        snap_path = download_one(job)
        if snap_path is None:
            failed.append(job["repo_id"] + (f":{job['allow_patterns'][0]}" if job.get("allow_patterns") else ""))
            continue
        if "config_key" not in job:
            continue  # tokenizer-style job — file-staging only, no config patch
        if "config_join" in job:
            updates[job["config_key"]] = snap_path / job["config_join"]
        else:
            updates[job["config_key"]] = snap_path

    patch_config(updates)

    if failed:
        print("\nFailed downloads:")
        for f in failed:
            print(f"  - {f}")
        return 1
    print("\nAll downloads complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
