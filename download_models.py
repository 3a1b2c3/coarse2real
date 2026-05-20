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
CONFIG_PATH = REPO_ROOT / "inference" / "config_1gpu.json"

JOBS = [
    {
        "repo_id": "Wan-AI/Wan2.1-T2V-14B",
        "allow_patterns": None,
        "config_key": "base_model_dir",
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
    print(f"=== {repo_id}" + (f"   ({job['allow_patterns'][0]})" if job.get("allow_patterns") else "") + " ===")
    try:
        path = snapshot_download(
            repo_id=repo_id,
            allow_patterns=job.get("allow_patterns"),
        )
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
    if not CONFIG_PATH.exists():
        print(f"\nConfig not found at {CONFIG_PATH}; skipping patch.")
        return
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    changed = []
    for key, value in updates.items():
        old = cfg.get(key)
        new = str(value).replace("\\", "/")
        if old != new:
            cfg[key] = new
            changed.append((key, old, new))
    if not changed:
        print("\nConfig already up to date.")
        return
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    print(f"\nPatched {CONFIG_PATH}:")
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
