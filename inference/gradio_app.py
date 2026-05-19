import argparse
import hashlib
import json
import os
import random
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import gradio as gr
import torch

try:
    from inference.inference_utils import (
        InferenceConfig,
        load_config,
    )
except ModuleNotFoundError:
    from inference_utils import (  # type: ignore
        InferenceConfig,
        load_config,
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
INFERENCE_DIR = REPO_ROOT / "inference"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "gradio_runs"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
DEMO_CSS = """
#control-video-source-tabs {
  border-bottom: 1px solid var(--border-color-primary, #e5e7eb);
  margin-bottom: 14px;
}
#control-video-source-tabs .wrap,
#control-video-source-tabs .radio-group {
  align-items: flex-end;
  gap: 22px;
}
#control-video-source-tabs label {
  background: transparent;
  border: 0;
  border-radius: 0;
  box-shadow: none;
  color: var(--body-text-color-subdued, #6b7280);
  margin: 0;
  padding: 0 0 9px 0;
}
#control-video-source-tabs label:has(input:checked) {
  border-bottom: 2px solid #f97316;
  color: #f97316;
}
#control-video-source-tabs input[type="radio"] {
  display: none;
}
#uploaded-control-video button[aria-label*="ownload"],
#uploaded-control-video a[download] {
  display: none !important;
}
"""


def _gradio_prefers_launch_css() -> bool:
    try:
        return int(str(gr.__version__).split(".", 1)[0]) >= 6
    except (AttributeError, TypeError, ValueError):
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local Gradio app for C2R / WAN inference.")
    parser.add_argument(
        "--config",
        default=str(INFERENCE_DIR / "config_multigpu_usp.json"),
        help="Fixed inference config used by the demo UI.",
    )
    parser.add_argument(
        "--examples-dir",
        default=str(INFERENCE_DIR / "control_videos"),
        help="Directory containing example control videos shown in the demo UI.",
    )
    parser.add_argument(
        "--gpu-ids",
        default="",
        help="Optional CUDA_VISIBLE_DEVICES value used by Gradio subprocess launches, for example 0 or 0,1,2,3.",
    )
    parser.add_argument(
        "--nproc-per-node",
        type=int,
        default=None,
        help="Optional USP process count. Defaults to the detected visible GPU count.",
    )
    parser.add_argument(
        "--server-name",
        default=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        help="Gradio bind address. Defaults to localhost only. Set GRADIO_SERVER_NAME=0.0.0.0 to bind externally.",
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        help="Gradio port.",
    )
    return parser.parse_args()


def sanitize_slug(value: str, fallback: str = "run") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", (value or "").strip()).strip("._-")
    return cleaned or fallback


def _expand_gpu_ids_text(gpu_ids_text: str) -> list[str]:
    ids: list[str] = []
    for raw_token in gpu_ids_text.split(","):
        token = raw_token.strip()
        if not token:
            continue
        match = re.fullmatch(r"(\d+)-(\d+)", token)
        if match:
            start = int(match.group(1))
            end = int(match.group(2))
            step = 1 if end >= start else -1
            ids.extend(str(index) for index in range(start, end + step, step))
        else:
            ids.append(token)
    return ids


def _count_gpu_list(gpu_ids_text: str | None) -> int:
    value = (gpu_ids_text or "").strip()
    if not value or value in {"NoDevFiles", "-1"}:
        return 0
    return len(_expand_gpu_ids_text(value))


def _scheduler_cuda_visible_devices(env: dict[str, str]) -> str | None:
    for key in ("SLURM_STEP_GPUS", "SLURM_JOB_GPUS"):
        value = (env.get(key) or "").strip()
        if not value:
            continue
        if any(marker in value for marker in (":", "[", "]", "(", ")")):
            continue
        ids = _expand_gpu_ids_text(value)
        if ids:
            return ",".join(ids)
    return None


def _scheduler_gpu_count(env: dict[str, str]) -> int:
    scheduler_visible = _scheduler_cuda_visible_devices(env)
    if scheduler_visible:
        return _count_gpu_list(scheduler_visible)

    for key in ("SLURM_GPUS_ON_NODE", "SLURM_GPUS_PER_NODE", "PBS_NUM_GPUS"):
        value = (env.get(key) or "").strip()
        if not value:
            continue
        if value.isdigit():
            return int(value)
        match = re.search(r":(\d+)(?:\(|$)", value)
        if match:
            return int(match.group(1))

    gpu_file = env.get("PBS_GPUFILE")
    if gpu_file:
        try:
            return len({line.strip() for line in Path(gpu_file).read_text(encoding="utf-8").splitlines() if line.strip()})
        except OSError:
            return 0
    return 0


def _maybe_export_scheduler_cuda_visible_devices(env: dict[str, str]) -> None:
    if env.get("CUDA_VISIBLE_DEVICES"):
        return
    scheduler_visible = _scheduler_cuda_visible_devices(env)
    if scheduler_visible:
        env["CUDA_VISIBLE_DEVICES"] = scheduler_visible


def infer_default_dp_nproc(path: Path, parallel_mode: str) -> int:
    if parallel_mode not in {"dp", "usp"}:
        return 2
    visible_count = _count_gpu_list(os.environ.get("CUDA_VISIBLE_DEVICES"))
    if visible_count:
        return max(visible_count, 2)
    scheduler_count = _scheduler_gpu_count(os.environ)
    if scheduler_count:
        return max(scheduler_count, 2)
    if torch.cuda.is_available():
        return max(torch.cuda.device_count(), 2)
    match = re.search(r"(\d+)gpu", path.stem.lower())
    if match:
        return max(int(match.group(1)), 2)
    return 2


def relative_label(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)

def write_prompts_file(prompts: list[str], destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(prompts) + "\n", encoding="utf-8")
    return destination


def unique_path(directory: Path, file_name: str) -> Path:
    candidate = directory / file_name
    if not candidate.exists():
        return candidate
    stem = Path(file_name).stem
    suffix = Path(file_name).suffix
    counter = 1
    while True:
        candidate = directory / f"{stem}_{counter:03d}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1

def normalize_gpu_ids(gpu_ids_text: str) -> list[str]:
    return _expand_gpu_ids_text(gpu_ids_text)


def load_inference_preset(path: Path) -> InferenceConfig:
    return load_config(str(path))


def discover_control_videos(examples_dir: str | Path) -> dict[str, Path]:
    root = Path(examples_dir)
    if not root.is_absolute():
        root = (REPO_ROOT / root).resolve()
    if not root.is_dir():
        return {}

    videos = sorted(
        path.resolve()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    examples: dict[str, Path] = {}
    for path in videos:
        try:
            label = str(path.relative_to(root))
        except ValueError:
            label = path.name
        examples[label] = path
    return examples


def preview_example_video(example_label: str | None, examples_dir: str) -> str | None:
    if not example_label:
        return None
    return str(discover_control_videos(examples_dir).get(example_label) or "") or None


def control_video_source_ui_updates(control_video_source: str):
    use_example = control_video_source == "Try An Example"
    return gr.update(visible=use_example), gr.update(visible=not use_example)


def _uploaded_video_path(uploaded_video: Any) -> Path | None:
    if uploaded_video is None:
        return None
    if isinstance(uploaded_video, str):
        path = Path(uploaded_video)
        return path if path.is_file() else None
    if isinstance(uploaded_video, dict):
        for key in ("path", "name"):
            value = uploaded_video.get(key)
            if value:
                path = Path(value)
                if path.is_file():
                    return path
    name = getattr(uploaded_video, "name", None)
    if name:
        path = Path(name)
        if path.is_file():
            return path
    return None


def _single_prompt(prompt_text: str | None) -> str:
    prompt = " ".join(line.strip() for line in (prompt_text or "").splitlines() if line.strip())
    if not prompt:
        raise ValueError("Please enter one prompt before running inference.")
    return prompt


def _resolve_seed(seed: Any) -> int:
    if seed is None:
        return random.randint(0, 2**31 - 1)
    if isinstance(seed, str) and not seed.strip():
        return random.randint(0, 2**31 - 1)
    return int(seed)


def _stage_demo_control_video(
    example_label: str | None,
    uploaded_video: Any,
    examples_dir: str,
    destination_dir: Path,
) -> tuple[Path, str]:
    uploaded_path = _uploaded_video_path(uploaded_video)
    source_label = "uploaded"
    source_path: Path | None = uploaded_path

    if source_path is None:
        examples = discover_control_videos(examples_dir)
        if example_label:
            source_path = examples.get(example_label)
            source_label = f"example:{example_label}"
        if source_path is None and examples:
            first_label, first_path = next(iter(examples.items()))
            source_path = first_path
            source_label = f"example:{first_label}"

    if source_path is None:
        raise ValueError("Please upload one control video or select one example control video.")
    if source_path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError(f"Unsupported control video format: {source_path.suffix}")

    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = unique_path(destination_dir, f"{sanitize_slug(source_path.stem, 'control')}{source_path.suffix.lower()}")
    shutil.copy2(source_path, destination)
    return destination.resolve(), source_label


def _demo_nproc_per_node(gpu_ids_text: str, explicit_nproc: int | None, config_path: Path) -> int:
    if explicit_nproc is not None and explicit_nproc > 0:
        return explicit_nproc
    gpu_count = _count_gpu_list(gpu_ids_text)
    if gpu_count:
        return max(gpu_count, 2)
    return infer_default_dp_nproc(config_path, "usp")


def _demo_info_markdown(cfg: InferenceConfig, config_path: Path, examples_dir: Path) -> str:
    clip_seconds = cfg.num_frames / cfg.control_video_target_fps if cfg.control_video_target_fps else 0.0
    dino_path = cfg.dino_model_path or "HF cache / facebook/dinov3-vitb16-pretrain-lvd1689m"
    return "\n".join(
        [
            "### Demo Setup",
            f"- Default initial config loaded from: `{relative_label(config_path)}`",
            "- Model: `14B` C2R, APG guidance",
            f"- Output format: `{cfg.width}x{cfg.height}`, `{cfg.num_frames}` frames, `{cfg.fps}` fps",
            f"- Control preprocessing: enabled, `{cfg.control_video_target_fps}` fps, about `{clip_seconds:.1f}` seconds per clip",
            f"- DINOv3 backbone: `{dino_path}`",
            f"- Example videos dir: `{relative_label(examples_dir)}`",
        ]
    )


def _resolve_app_path(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    return resolved.resolve()


def _gradio_allowed_paths(config_path: str | Path, examples_dir: str | Path) -> list[str]:
    roots = [
        REPO_ROOT.resolve(),
        DEFAULT_OUTPUT_DIR.resolve(),
        _resolve_app_path(examples_dir),
    ]

    try:
        cfg = load_inference_preset(_resolve_app_path(config_path))
    except Exception:
        cfg = None

    if cfg is not None:
        output_root = Path(cfg.output_dir)
        if not output_root.is_absolute():
            output_root = REPO_ROOT / output_root
        output_root.mkdir(parents=True, exist_ok=True)
        roots.append(output_root.resolve())

        if cfg.control_videos_dir:
            control_root = Path(cfg.control_videos_dir)
            if not control_root.is_absolute():
                control_root = REPO_ROOT / control_root
            if control_root.exists():
                roots.append(control_root.resolve())

    return list(dict.fromkeys(str(root) for root in roots))


def write_resolved_config(run_cfg: InferenceConfig, run_dir: Path) -> Path:
    config_path = run_dir / "resolved_config.json"
    config_path.write_text(json.dumps(asdict(run_cfg), indent=2, ensure_ascii=False), encoding="utf-8")
    return config_path


def load_run_records(run_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for metadata_path in sorted(run_dir.glob("results_rank*.jsonl")):
        for line in metadata_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            records.append(json.loads(line))
    records.sort(key=lambda record: (record["prompt_id"], -1 if record["control_video_id"] is None else record["control_video_id"]))
    return records


def build_result_caption(run_label: str, run_cfg: InferenceConfig, record: dict[str, Any]) -> str:
    caption_parts = [
        f"run={run_label}",
        f"prompt={record['prompt_id']}",
        f"seed={record.get('seed', run_cfg.seed)}",
        f"guidance={run_cfg.guidance_mode}",
        f"cv_scale={run_cfg.control_video_scale}",
        f"text_scale={run_cfg.text_scale}",
    ]
    control_video_path = record.get("control_video_path")
    if control_video_path is not None:
        caption_parts.append(f"control={Path(control_video_path).name}")
    return " | ".join(caption_parts)


def finalize_run_records(
    run_label: str,
    run_cfg: InferenceConfig,
    records: list[dict[str, Any]],
) -> tuple[list[tuple[str, str]], list[str], list[dict[str, Any]]]:
    gallery_items: list[tuple[str, str]] = []
    output_files: list[str] = []
    normalized_records: list[dict[str, Any]] = []
    for record in records:
        caption = build_result_caption(run_label, run_cfg, record)
        output_path = str(Path(record["output"]).resolve())
        gallery_items.append((output_path, caption))
        output_files.append(output_path)
        normalized_records.append({**record, "output": output_path, "caption": caption})
    return gallery_items, output_files, normalized_records


def read_enhanced_prompt_text(run_dir: Path) -> str:
    prompts_path = run_dir / "prompt_enhancement" / "enhanced_prompts.jsonl"
    if not prompts_path.exists():
        return ""

    enhanced_prompts: list[str] = []
    try:
        for line in prompts_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            enhanced_prompt = str(record.get("enhanced_prompt") or "").strip()
            if enhanced_prompt:
                enhanced_prompts.append(enhanced_prompt)
    except (OSError, json.JSONDecodeError):
        return ""

    return "\n\n".join(dict.fromkeys(enhanced_prompts))


def _demo_outputs(
    summary: str,
    log_text: str,
    enhanced_prompt: str = "",
    gallery_items: list[tuple[str, str]] | None = None,
    manifest: dict[str, Any] | None = None,
    manifest_path: str | None = None,
    downloads: list[str] | None = None,
):
    return (
        summary,
        log_text,
        enhanced_prompt,
        gallery_items or [],
        manifest or {},
        manifest_path,
        downloads or [],
    )


def _stream_logged_subprocess(
    command: list[str],
    log_path: Path,
    progress: gr.Progress,
    start_desc: str,
    end_desc: str,
    env: dict[str, str],
):
    progress(0, desc=start_desc)
    process = subprocess.Popen(
        command,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )

    log_lines: list[str] = []
    last_emit = 0.0
    with log_path.open("w", encoding="utf-8") as log_file:
        assert process.stdout is not None
        for line in process.stdout:
            log_file.write(line)
            log_file.flush()
            text = line.rstrip()
            if not text:
                continue
            log_lines.append(text)
            if len(log_lines) > 200:
                log_lines = log_lines[-200:]
            now = time.monotonic()
            if now - last_emit >= 1.0:
                last_emit = now
                yield "\n".join(log_lines)

    return_code = process.wait()
    tail_log = "\n".join(log_lines)
    if return_code != 0:
        raise RuntimeError(
            f"Subprocess failed with exit code {return_code}.\n\nLast log lines:\n{tail_log}"
        )

    progress(1, desc=end_desc)
    yield tail_log


def _usp_timeout_defaults(nproc_per_node: int) -> dict[str, str]:
    if nproc_per_node >= 8:
        return {
            "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC": "300",
            "C2R_USP_PG_TIMEOUT_SEC": "600",
            "C2R_NCCL_SMOKE_TIMEOUT_SEC": "60",
            "C2R_NCCL_PREFLIGHT_HEARTBEAT_TIMEOUT_SEC": "90",
            "C2R_NCCL_PREFLIGHT_HARD_TIMEOUT_SEC": "180",
        }
    if nproc_per_node >= 4:
        return {
            "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC": "240",
            "C2R_USP_PG_TIMEOUT_SEC": "420",
            "C2R_NCCL_SMOKE_TIMEOUT_SEC": "30",
            "C2R_NCCL_PREFLIGHT_HEARTBEAT_TIMEOUT_SEC": "60",
            "C2R_NCCL_PREFLIGHT_HARD_TIMEOUT_SEC": "120",
        }
    return {
        "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC": "120",
        "C2R_USP_PG_TIMEOUT_SEC": "180",
        "C2R_NCCL_SMOKE_TIMEOUT_SEC": "8",
        "C2R_NCCL_PREFLIGHT_HEARTBEAT_TIMEOUT_SEC": "10",
        "C2R_NCCL_PREFLIGHT_HARD_TIMEOUT_SEC": "20",
    }


def _apply_usp_env_defaults(env: dict[str, str], nproc_per_node: int) -> dict[str, str]:
    env.setdefault("TORCHDYNAMO_DISABLE", "1")
    env.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    env.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
    env.setdefault("TORCH_FR_BUFFER_SIZE", "20000")
    env.setdefault("TORCH_NCCL_DUMP_ON_TIMEOUT", "1")
    for key, value in _usp_timeout_defaults(nproc_per_node).items():
        env.setdefault(key, value)
    env.setdefault("C2R_NCCL_PREFLIGHT_TIMEOUT_POLICY", "continue")
    return env


def _reset_nccl_transport_overrides(env: dict[str, str]) -> None:
    env.pop("NCCL_P2P_DISABLE", None)
    env.pop("NCCL_P2P_LEVEL", None)
    env.pop("NCCL_SHM_DISABLE", None)


def _apply_nccl_transport_mode(env: dict[str, str], mode: str) -> None:
    _reset_nccl_transport_overrides(env)
    if mode == "default":
        return
    if mode == "pix":
        env["NCCL_P2P_LEVEL"] = "PIX"
        return
    if mode == "safe":
        env["NCCL_P2P_DISABLE"] = "1"
        return
    raise ValueError(f"Unknown NCCL transport mode: {mode}")


def _build_nccl_cache_file(nproc_per_node: int, env: dict[str, str]) -> Path:
    runtime_root = Path(env.get("C2R_RUNTIME_DIR") or tempfile.gettempdir()) / "c2r_runtime" / "nccl_transport_cache"
    runtime_root.mkdir(parents=True, exist_ok=True)
    host_id = socket.gethostname() or "unknown-host"
    gpu_id = env.get("CUDA_VISIBLE_DEVICES", "all")
    key_source = f"{host_id}|{gpu_id}|{nproc_per_node}"
    key_hash = hashlib.sha1(key_source.encode("utf-8")).hexdigest()
    return runtime_root / f"{key_hash}.json"


def _save_cached_nccl_mode(cache_path: Path, label: str, env: dict[str, str]) -> None:
    cache_path.write_text(
        json.dumps(
            {
                "label": label,
                "NCCL_P2P_DISABLE": env.get("NCCL_P2P_DISABLE", ""),
                "NCCL_P2P_LEVEL": env.get("NCCL_P2P_LEVEL", ""),
                "NCCL_SHM_DISABLE": env.get("NCCL_SHM_DISABLE", ""),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _load_cached_nccl_mode(cache_path: Path, env: dict[str, str]) -> str | None:
    if not cache_path.exists():
        return None
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        cache_path.unlink(missing_ok=True)
        return None

    label = payload.get("label")
    if not label:
        cache_path.unlink(missing_ok=True)
        return None

    _reset_nccl_transport_overrides(env)
    for key in ("NCCL_P2P_DISABLE", "NCCL_P2P_LEVEL", "NCCL_SHM_DISABLE"):
        value = payload.get(key)
        if value:
            env[key] = str(value)
    return str(label)


def _filter_preflight_output(output: str) -> str:
    interesting: list[str] = []
    for line in output.splitlines():
        if any(
            token in line
            for token in (
                "Running NCCL preflight on",
                "NCCL preflight passed.",
                "Watchdog",
                "timed out",
                "failed",
                "ProcessGroupNCCL",
                "Warning",
            )
        ):
            if "Traceback" in line or "SignalException" in line:
                continue
            interesting.append(line)
    return "\n".join(interesting)


def _run_nccl_preflight(
    label: str,
    nproc_per_node: int,
    env: dict[str, str],
    progress: gr.Progress,
) -> tuple[bool, str, str]:
    summary_lines = [
        f"Running NCCL preflight ({label}) with NPROC_PER_NODE={nproc_per_node}...",
        "NCCL transport overrides: "
        f"NCCL_P2P_DISABLE={env.get('NCCL_P2P_DISABLE', '0')} "
        f"NCCL_P2P_LEVEL={env.get('NCCL_P2P_LEVEL', 'auto')} "
        f"NCCL_SHM_DISABLE={env.get('NCCL_SHM_DISABLE', '0')}",
    ]
    progress(0, desc=f"NCCL preflight: {label}")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={nproc_per_node}",
        "-m",
        "inference.nccl_smoke_test",
    ]
    process = subprocess.Popen(
        command,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={
            **env,
            "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC": env.get("C2R_NCCL_PREFLIGHT_HEARTBEAT_TIMEOUT_SEC", "10"),
            "C2R_NCCL_SMOKE_TIMEOUT_SEC": env.get("C2R_NCCL_SMOKE_TIMEOUT_SEC", "8"),
        },
    )

    hard_timeout_sec = int(env.get("C2R_NCCL_PREFLIGHT_HARD_TIMEOUT_SEC", "20"))
    try:
        stdout, _ = process.communicate(timeout=hard_timeout_sec)
        status = process.returncode
        timed_out = False
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            stdout, _ = process.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, _ = process.communicate()
        status = 124
        timed_out = True

    filtered = _filter_preflight_output(stdout or "")
    if status == 0:
        if filtered:
            summary_lines.append(filtered)
        return True, "\n".join(summary_lines), "passed"

    timeout_output = bool(re.search(r"timed out|timeout|Watchdog", stdout or "", re.IGNORECASE))
    if timed_out or status in {124, 137, 143} or timeout_output:
        summary_lines.append(
            f"NCCL preflight ({label}) timed out after {hard_timeout_sec}s. Trying next transport mode..."
        )
        result = "timeout"
    else:
        summary_lines.append(f"NCCL preflight ({label}) failed with exit code {status}. Trying next transport mode...")
        result = "failed"
    if filtered:
        summary_lines.append(filtered)
    return False, "\n".join(summary_lines), result


def _prepare_usp_env(
    nproc_per_node: int,
    gpu_ids_text: str,
    progress: gr.Progress,
) -> tuple[dict[str, str], str]:
    env = _apply_usp_env_defaults(os.environ.copy(), nproc_per_node)
    gpu_ids = normalize_gpu_ids(gpu_ids_text)
    if gpu_ids:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
    else:
        _maybe_export_scheduler_cuda_visible_devices(env)

    if env.get("C2R_NCCL_SAFE_MODE") == "1":
        env.setdefault("NCCL_P2P_DISABLE", "1")
    if env.get("C2R_NCCL_SHM_FALLBACK") == "1":
        env.setdefault("NCCL_SHM_DISABLE", "1")

    if env.get("C2R_SKIP_NCCL_PREFLIGHT") == "1":
        return env, "Skipping NCCL preflight because C2R_SKIP_NCCL_PREFLIGHT=1."

    manual_override = any(env.get(key) for key in ("NCCL_P2P_DISABLE", "NCCL_P2P_LEVEL", "NCCL_SHM_DISABLE"))
    cache_path = _build_nccl_cache_file(nproc_per_node, env)
    logs: list[str] = []
    timeout_seen = False
    non_timeout_failure = False

    if manual_override:
        success, summary, result = _run_nccl_preflight("manual overrides", nproc_per_node, env, progress)
        logs.append(summary)
        if success:
            return env, "\n".join(logs)
        if result == "timeout" and env.get("C2R_NCCL_PREFLIGHT_TIMEOUT_POLICY") == "continue":
            logs.append(
                "NCCL preflight timed out with manual transport overrides; continuing because "
                "C2R_NCCL_PREFLIGHT_TIMEOUT_POLICY=continue."
            )
            return env, "\n".join(logs)
        else:
            raise RuntimeError(summary)

    cached_label = _load_cached_nccl_mode(cache_path, env)
    if cached_label:
        success, summary, result = _run_nccl_preflight(cached_label, nproc_per_node, env, progress)
        logs.append(f"Trying cached NCCL transport mode: {cached_label}")
        logs.append(summary)
        if success:
            logs.append(f"Selected NCCL transport mode: {cached_label}")
            return env, "\n".join(logs)
        if result == "timeout":
            timeout_seen = True
        else:
            non_timeout_failure = True
        cache_path.unlink(missing_ok=True)

    for label, mode_key, cache_label in (
        ("default", "default", "default"),
        ("NCCL_P2P_LEVEL=PIX", "pix", "NCCL_P2P_LEVEL=PIX"),
        ("NCCL_P2P_DISABLE=1", "safe", "NCCL_P2P_DISABLE=1"),
    ):
        _apply_nccl_transport_mode(env, mode_key)
        success, summary, result = _run_nccl_preflight(label, nproc_per_node, env, progress)
        logs.append(summary)
        if success:
            _save_cached_nccl_mode(cache_path, cache_label, env)
            logs.append(f"Selected NCCL transport mode: {cache_label}")
            return env, "\n".join(logs)
        if result == "timeout":
            timeout_seen = True
        else:
            non_timeout_failure = True

    if timeout_seen and not non_timeout_failure and env.get("C2R_NCCL_PREFLIGHT_TIMEOUT_POLICY") == "continue":
        _apply_nccl_transport_mode(env, "default")
        logs.append("NCCL preflight timed out for all transport modes but did not report a concrete NCCL error.")
        logs.append("Continuing with default NCCL transport because C2R_NCCL_PREFLIGHT_TIMEOUT_POLICY=continue.")
        logs.append("Set C2R_NCCL_PREFLIGHT_TIMEOUT_POLICY=fail or C2R_SKIP_NCCL_PREFLIGHT=1 to change this behavior.")
        return env, "\n".join(logs)

    raise RuntimeError(
        "\n".join(
            [
                *logs,
                "NCCL preflight failed for default, NCCL_P2P_LEVEL=PIX, and NCCL_P2P_DISABLE=1.",
            ]
        )
    )


def run_app(
    prompt_text: str,
    control_video_source: str,
    example_video_label: str | None,
    uploaded_control_video: Any,
    enhance_prompt: bool,
    inference_mode: str,
    text_scale: float,
    control_video_scale: float,
    seed: float,
    config_path_text: str,
    examples_dir_text: str,
    gpu_ids_text: str,
    usp_nproc_per_node: int | None,
    progress=gr.Progress(track_tqdm=True),
):
    status_lines: list[str] = []

    def set_status(message: str) -> str:
        status_lines.append(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
        return "\n".join(status_lines)

    try:
        yield _demo_outputs("Preparing generation...", set_status("Validating prompt and selected control video."))

        config_path = Path(config_path_text)
        if not config_path.is_absolute():
            config_path = (REPO_ROOT / config_path).resolve()
        examples_dir = Path(examples_dir_text)
        if not examples_dir.is_absolute():
            examples_dir = (REPO_ROOT / examples_dir).resolve()

        prompt = _single_prompt(prompt_text)
        base_cfg = load_inference_preset(config_path)
        session_slug = sanitize_slug(config_path.stem, "c2r-demo")
        output_root = Path(base_cfg.output_dir)
        if not output_root.is_absolute():
            output_root = REPO_ROOT / output_root
        session_dir = output_root.resolve() / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{session_slug}"
        run_dir = session_dir / "generation"
        inputs_dir = session_dir / "inputs"
        control_dir = inputs_dir / "control_video"
        run_dir.mkdir(parents=True, exist_ok=True)

        yield _demo_outputs(
            f"Preparing session: `{relative_label(session_dir)}`",
            set_status("Copying the control video into the run folder."),
        )
        use_upload = control_video_source == "Upload My Video"
        if use_upload and _uploaded_video_path(uploaded_control_video) is None:
            raise ValueError("Please upload one control video before running inference.")
        control_video_path, control_source_label = _stage_demo_control_video(
            example_label=None if use_upload else example_video_label,
            uploaded_video=uploaded_control_video if use_upload else None,
            examples_dir=str(examples_dir),
            destination_dir=control_dir,
        )

        run_cfg = deepcopy(base_cfg)
        run_cfg.guidance_mode = "apg"
        run_cfg.prompts_file = str(write_prompts_file([prompt], inputs_dir / "prompt.txt"))
        run_cfg.control_videos_dir = str(control_dir)
        run_cfg.output_dir = str(run_dir)
        run_cfg.preprocess_control_videos = True
        run_cfg.control_video_preprocess_dir = str(session_dir / "preprocessed_control_videos")
        run_cfg.control_video_target_fps = 16
        run_cfg.control_short_video_strategy = "pad"
        run_cfg.prompt_enhancement_mode = "enhanced" if enhance_prompt else "off"
        run_cfg.text_scale = float(text_scale)
        run_cfg.control_video_scale = float(control_video_scale)
        run_cfg.seed = _resolve_seed(seed)
        run_cfg.parallel_mode = inference_mode

        resolved_config = write_resolved_config(run_cfg, run_dir)
        yield _demo_outputs(
            f"Prepared inputs for `{control_video_path.name}`",
            set_status("Resolved the fixed model config and wrote the one-prompt run config."),
        )

        if inference_mode == "single-gpu":
            gpu_ids = normalize_gpu_ids(gpu_ids_text)
            log_path = run_dir / "single_gpu_launcher.log"
            command = [
                sys.executable,
                "-m",
                "inference.run_inference",
                "--config",
                str(resolved_config),
            ]
            env = os.environ.copy()
            if gpu_ids:
                env["CUDA_VISIBLE_DEVICES"] = gpu_ids[0]
            else:
                _maybe_export_scheduler_cuda_visible_devices(env)
            header = "\n".join(
                [
                    "Execution mode: single-gpu",
                    f"GPU id: {gpu_ids[0] if gpu_ids else env.get('CUDA_VISIBLE_DEVICES', 'default visible GPU')}",
                    f"Command: {' '.join(command)}",
                    f"Log file: {log_path}",
                    "",
                ]
            )
            start_desc = "Launching single-GPU generation"
            end_desc = "Single-GPU generation finished"
        elif inference_mode == "usp":
            nproc_per_node = _demo_nproc_per_node(gpu_ids_text, usp_nproc_per_node, config_path)
            yield _demo_outputs(
                f"Preparing USP launch for `{nproc_per_node}` process(es)",
                set_status("Running/validating USP NCCL transport before model loading."),
            )
            env, preflight_log = _prepare_usp_env(nproc_per_node, gpu_ids_text, progress)
            gpu_ids = normalize_gpu_ids(gpu_ids_text)
            log_path = run_dir / "usp_launcher.log"
            command = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc_per_node={nproc_per_node}",
                "-m",
                "inference.run_inference",
                "--config",
                str(resolved_config),
            ]
            header = "\n".join(
                [
                    "Execution mode: usp",
                    f"Processes launched: {nproc_per_node}",
                    f"GPU ids: {','.join(gpu_ids) if gpu_ids else env.get('CUDA_VISIBLE_DEVICES', 'default visible GPUs')}",
                    f"Command: {' '.join(command)}",
                    f"Log file: {log_path}",
                    "",
                    preflight_log,
                    "",
                ]
            )
            start_desc = "Launching USP generation"
            end_desc = "USP generation finished"
        else:
            raise ValueError("The demo app supports only single-gpu and usp inference modes.")

        yield _demo_outputs(
            f"Generation running for `{control_video_path.name}`",
            set_status("Launching inference. The live log below will update while the model runs.") + "\n\n" + header,
        )
        final_tail_log = ""
        enhanced_prompt_text = ""
        enhanced_prompt_reported = False
        for tail_log in _stream_logged_subprocess(
            command=command,
            log_path=log_path,
            progress=progress,
            start_desc=start_desc,
            end_desc=end_desc,
            env=env,
        ):
            final_tail_log = tail_log
            latest_enhanced_prompt = read_enhanced_prompt_text(run_dir)
            if latest_enhanced_prompt:
                enhanced_prompt_text = latest_enhanced_prompt
                if not enhanced_prompt_reported:
                    set_status("Prompt enhancement finished; showing the enhanced prompt.")
                    enhanced_prompt_reported = True
            yield _demo_outputs(
                f"Generation running for `{control_video_path.name}`",
                "\n".join(status_lines) + "\n\n" + header + tail_log,
                enhanced_prompt_text,
            )

        run_records = load_run_records(run_dir)
        if not run_records:
            raise RuntimeError("Inference finished, but no results_rank*.jsonl records were found.")

        gallery_items, output_files, run_results = finalize_run_records(
            run_label="generation",
            run_cfg=run_cfg,
            records=run_records,
        )
        prompt_enhancement_artifacts = [
            str(path)
            for path in (
                run_dir / "prompt_enhancement" / "settings.json",
                run_dir / "prompt_enhancement" / "video_descriptions.json",
                run_dir / "prompt_enhancement" / "enhanced_prompts.jsonl",
            )
            if path.exists()
        ]
        enhanced_prompt_text = read_enhanced_prompt_text(run_dir)
        manifest = {
            "created_at": datetime.now().isoformat(),
            "app": "c2r-demo",
            "config": str(config_path),
            "session_dir": str(session_dir),
            "prompt": prompt,
            "enhanced_prompt": enhanced_prompt_text or None,
            "control_video_source_ui": control_video_source,
            "control_video": str(control_video_path),
            "control_source": control_source_label,
            "prompt_enhancement_mode": run_cfg.prompt_enhancement_mode,
            "parallel_mode": inference_mode,
            "seed": run_cfg.seed,
            "text_scale": run_cfg.text_scale,
            "control_video_scale": run_cfg.control_video_scale,
            "resolved_config": str(resolved_config),
            "launcher_log": str(log_path),
            "prompt_enhancement_artifacts": prompt_enhancement_artifacts,
            "results": run_results,
        }
        manifest_path = session_dir / "session_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

        summary = "\n".join(
            [
                f"Session directory: `{relative_label(session_dir)}`",
                f"Control video: `{control_video_path.name}`",
                f"Generated clips: `{len(gallery_items)}`",
                f"Prompt enhancement: `{run_cfg.prompt_enhancement_mode}`",
                f"Inference mode: `{inference_mode}`",
                f"Seed: `{run_cfg.seed}`",
            ]
        )
        downloads = [
            str(manifest_path),
            str(resolved_config),
            str(log_path),
            str(control_video_path),
            *output_files,
            *prompt_enhancement_artifacts,
        ]
        yield _demo_outputs(
            summary,
            set_status("Generation finished successfully.") + "\n\n" + header + final_tail_log,
            enhanced_prompt_text,
            gallery_items,
            manifest,
            str(manifest_path),
            downloads,
        )
    except Exception as exc:
        error_log = set_status(f"Generation failed: {type(exc).__name__}: {exc}")
        yield _demo_outputs("Generation failed.", error_log)


def build_app(
    config_path: str | Path | None = None,
    examples_dir: str | Path | None = None,
    gpu_ids_text: str = "",
    usp_nproc_per_node: int | None = None,
) -> gr.Blocks:
    config_path = Path(config_path or INFERENCE_DIR / "config_multigpu_usp.json")
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()
    examples_dir = Path(examples_dir or INFERENCE_DIR / "control_videos")
    if not examples_dir.is_absolute():
        examples_dir = (REPO_ROOT / examples_dir).resolve()

    cfg = load_inference_preset(config_path)
    cfg.guidance_mode = "apg"
    cfg.preprocess_control_videos = True
    cfg.control_video_target_fps = 16
    cfg.control_short_video_strategy = "pad"
    examples = discover_control_videos(examples_dir)
    example_labels = list(examples)
    default_example = example_labels[0] if example_labels else None
    default_preview = str(examples[default_example]) if default_example else None
    default_mode = "single-gpu"
    text_scale_max = max(15.0, float(cfg.text_scale) * 2.0)
    control_scale_max = max(2.5, float(cfg.control_video_scale) * 2.0)
    default_seed = int(cfg.seed)

    blocks_kwargs = {"title": "C2R Demo"}
    if not _gradio_prefers_launch_css():
        blocks_kwargs["css"] = DEMO_CSS

    with gr.Blocks(**blocks_kwargs) as demo:
        gr.Markdown(
            """
            # C2R Demo

            Choose or load one control video, write one prompt, and generate a controlled video with the C2R model.
            """
        )
        gr.Markdown(_demo_info_markdown(cfg, config_path, examples_dir))

        control_video_source = gr.Radio(
            choices=["Try An Example", "Upload My Video"],
            value="Try An Example",
            label="Select Input Control Video",
            elem_id="control-video-source-tabs",
        )
        with gr.Row():
            with gr.Column(visible=True) as example_input_group:
                example_video = gr.Dropdown(
                    choices=example_labels,
                    value=default_example,
                    label="Example Control Video",
                    interactive=bool(example_labels),
                )
                example_preview = gr.Video(
                    value=default_preview,
                    label="Example Preview",
                    interactive=False,
                )
            with gr.Column(visible=False) as upload_input_group:
                uploaded_control_video = gr.Video(label="Control Video Upload", elem_id="uploaded-control-video")

        prompt_text = gr.Textbox(
            label="Prompt",
            lines=5,
            placeholder=(
                "Describe the desired style, location, background, buildings, weather, "
                "time of day, and visual mood."
            ),
        )
        enhance_prompt = gr.Checkbox(
            value=cfg.prompt_enhancement_mode == "enhanced",
            label="Enhance my prompt",
            info="Uses the C2R prompt enhancer for better generations. This takes more time.",
        )

        with gr.Row():
            inference_mode = gr.Radio(
                choices=["single-gpu", "usp"],
                value=default_mode,
                label="Inference Mode",
            )
            text_scale = gr.Slider(
                minimum=1.0,
                maximum=text_scale_max,
                step=0.5,
                value=float(cfg.text_scale),
                label="Text Weight",
            )
            control_video_scale = gr.Slider(
                minimum=0.0,
                maximum=control_scale_max,
                step=0.1,
                value=float(cfg.control_video_scale),
                label="Control Video Weight",
            )
            seed = gr.Number(
                value=default_seed,
                precision=0,
                label="Seed",
            )

        with gr.Row():
            run_button = gr.Button("Run Generation", variant="primary")
            clear_log_button = gr.Button("Clear Log")

        summary = gr.Markdown(label="Summary")
        run_log = gr.Textbox(label="Run Log", lines=16)
        enhanced_prompt = gr.Textbox(label="Enhanced Prompt", lines=6, interactive=False)
        gallery = gr.Gallery(label="Generated Result", columns=1, height="640px", preview=True)
        manifest_json = gr.JSON(label="Run Manifest", visible=False)
        manifest_file = gr.File(label="Manifest")
        output_files = gr.File(label="Downloads", file_count="multiple")

        config_state = gr.State(str(config_path))
        examples_dir_state = gr.State(str(examples_dir))
        gpu_ids_state = gr.State(gpu_ids_text)
        nproc_state = gr.State(usp_nproc_per_node)

        example_video.change(
            fn=preview_example_video,
            inputs=[example_video, examples_dir_state],
            outputs=[example_preview],
        )
        control_video_source.change(
            fn=control_video_source_ui_updates,
            inputs=[control_video_source],
            outputs=[example_input_group, upload_input_group],
        )
        clear_log_button.click(
            fn=lambda: ("Ready for the next run.", "", "", [], {}, None, []),
            outputs=[summary, run_log, enhanced_prompt, gallery, manifest_json, manifest_file, output_files],
        )
        run_button.click(
            fn=run_app,
            inputs=[
                prompt_text,
                control_video_source,
                example_video,
                uploaded_control_video,
                enhance_prompt,
                inference_mode,
                text_scale,
                control_video_scale,
                seed,
                config_state,
                examples_dir_state,
                gpu_ids_state,
                nproc_state,
            ],
            outputs=[summary, run_log, enhanced_prompt, gallery, manifest_json, manifest_file, output_files],
        )

    return demo


if __name__ == "__main__":
    args = parse_args()
    demo = build_app(
        config_path=args.config,
        examples_dir=args.examples_dir,
        gpu_ids_text=args.gpu_ids,
        usp_nproc_per_node=args.nproc_per_node,
    )
    launch_kwargs = {
        "server_name": args.server_name,
        "server_port": args.server_port,
        "share": False,
        "inbrowser": False,
        "allowed_paths": _gradio_allowed_paths(args.config, args.examples_dir),
    }
    if _gradio_prefers_launch_css():
        launch_kwargs["css"] = DEMO_CSS

    demo.queue(default_concurrency_limit=1).launch(**launch_kwargs)
