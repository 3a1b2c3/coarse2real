import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

import imageio
import numpy as np


MANIFEST_VERSION = 1
FPS_TOLERANCE = 1e-2


@dataclass
class ControlVideoPrepConfig:
    output_dir: Path
    target_fps: int
    target_num_frames: int
    short_video_strategy: str = "pad"
    video_encoding_quality: int = 5


def _sanitize_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-") or "control"


def _source_signature(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _source_id(path: Path) -> str:
    payload = json.dumps(_source_signature(path), sort_keys=True).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:10]


def _artifact_dir(source_path: Path, config: ControlVideoPrepConfig) -> Path:
    stem = _sanitize_name(source_path.stem)
    return config.output_dir / f"{stem}__{_source_id(source_path)}"


def _manifest_path(source_path: Path, config: ControlVideoPrepConfig) -> Path:
    return _artifact_dir(source_path, config) / "manifest.json"


def _clip_output_path(artifact_dir: Path, clip_index: int) -> Path:
    return artifact_dir / f"clip_{clip_index:03d}.mp4"


def _probe_video(path: Path, fallback_fps: int) -> tuple[float, int]:
    reader = imageio.get_reader(str(path))
    try:
        meta = reader.get_meta_data()
        try:
            frame_count = int(reader.count_frames())
        except Exception:
            frame_count = int(meta.get("nframes", 0)) if meta.get("nframes") not in (None, float("inf")) else 0
            if frame_count <= 0:
                frame_count = sum(1 for _ in reader)
        fps = meta.get("fps")
        fps = float(fps) if fps not in (None, 0) else 0.0
        if fps <= 0:
            duration = meta.get("duration")
            if duration not in (None, 0) and frame_count > 0:
                fps = float(frame_count) / float(duration)
            else:
                fps = float(fallback_fps)
        return fps, frame_count
    finally:
        reader.close()


def _resampled_frame_count(source_frame_count: int, source_fps: float, target_fps: int) -> int:
    if source_frame_count <= 0:
        return 0
    return max(1, int(round(source_frame_count * float(target_fps) / float(source_fps))))


def _build_sample_indices(source_frame_count: int, source_fps: float, target_fps: int, output_frame_count: int) -> np.ndarray:
    target_positions = np.arange(output_frame_count, dtype=np.float64)
    source_positions = np.round(target_positions * float(source_fps) / float(target_fps)).astype(np.int64)
    return np.clip(source_positions, 0, source_frame_count - 1)


def _manifest_matches(manifest: dict, source_path: Path, config: ControlVideoPrepConfig) -> bool:
    if manifest.get("version") != MANIFEST_VERSION:
        return False
    if manifest.get("source") != _source_signature(source_path):
        return False
    if manifest.get("target_fps") != config.target_fps:
        return False
    if manifest.get("target_num_frames") != config.target_num_frames:
        return False
    if manifest.get("short_video_strategy") != config.short_video_strategy:
        return False
    clip_paths = [Path(path) for path in manifest.get("clip_paths", [])]
    if not clip_paths:
        return False
    return all(path.is_file() for path in clip_paths if str(path) != str(source_path.resolve()))


def _load_existing_manifest(source_path: Path, config: ControlVideoPrepConfig) -> dict | None:
    path = _manifest_path(source_path, config)
    if not path.is_file():
        return None
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not _manifest_matches(manifest, source_path, config):
        return None
    return manifest


def _wait_for_preprocessed_manifests(
    source_paths: List[Path],
    config: ControlVideoPrepConfig,
    timeout_seconds: float = 1800.0,
    poll_interval_seconds: float = 1.0,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    pending = {path.resolve() for path in source_paths}

    while pending:
        ready = {path for path in pending if _load_existing_manifest(path, config) is not None}
        pending -= ready
        if not pending:
            return
        if time.monotonic() >= deadline:
            missing = ", ".join(path.name for path in sorted(pending))
            raise TimeoutError(
                "Timed out waiting for rank 0 to finish preparing control video manifests. "
                f"Still missing: {missing}"
            )
        time.sleep(poll_interval_seconds)


def _write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_clip_frames(
    source_path: Path,
    source_frame_count: int,
    artifact_dir: Path,
    sample_indices: np.ndarray,
    config: ControlVideoPrepConfig,
    clip_count: int,
    pad_to_full_clip: bool,
) -> List[Path]:
    if clip_count <= 0:
        return []

    clip_paths = [_clip_output_path(artifact_dir, clip_index) for clip_index in range(clip_count)]
    writer = None
    reader = imageio.get_reader(str(source_path))
    target_num_frames = config.target_num_frames
    pointer = 0
    clip_index = 0
    frames_in_clip = 0
    last_frame = None

    def open_writer(index: int):
        return imageio.get_writer(
            str(clip_paths[index]),
            fps=config.target_fps,
            quality=config.video_encoding_quality,
        )

    try:
        writer = open_writer(clip_index)
        for source_index in range(source_frame_count):
            frame = np.array(reader.get_data(source_index))
            last_frame = frame
            while pointer < len(sample_indices) and sample_indices[pointer] == source_index:
                writer.append_data(frame)
                pointer += 1
                frames_in_clip += 1
                if frames_in_clip == target_num_frames:
                    writer.close()
                    writer = None
                    clip_index += 1
                    frames_in_clip = 0
                    if clip_index < clip_count:
                        writer = open_writer(clip_index)
            if pointer >= len(sample_indices):
                break

        if writer is not None and pad_to_full_clip and frames_in_clip > 0 and last_frame is not None:
            while frames_in_clip < target_num_frames:
                writer.append_data(last_frame)
                frames_in_clip += 1
    finally:
        reader.close()
        if writer is not None:
            writer.close()

    return clip_paths


def prepare_control_video(source_path: Path, config: ControlVideoPrepConfig, allow_create: bool = True) -> List[Path]:
    source_path = source_path.resolve()
    manifest = _load_existing_manifest(source_path, config)
    if manifest is not None:
        return [Path(path) for path in manifest["clip_paths"]]

    if not allow_create:
        raise FileNotFoundError(
            f"Preprocessed control manifest not found for {source_path}. "
            "Run preprocessing on rank 0 first."
        )

    if config.target_fps <= 0:
        raise ValueError("control_target_fps must be > 0.")
    if config.target_num_frames <= 0:
        raise ValueError("target_num_frames must be > 0.")
    if config.short_video_strategy not in {"pad", "error"}:
        raise ValueError("short_video_strategy must be 'pad' or 'error'.")

    artifact_dir = _artifact_dir(source_path, config)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    source_fps, source_frame_count = _probe_video(source_path, fallback_fps=config.target_fps)
    if source_frame_count <= 0:
        raise ValueError(f"Control video has no readable frames: {source_path}")
    target_frame_count = _resampled_frame_count(source_frame_count, source_fps, config.target_fps)

    uses_original = (
        source_frame_count == config.target_num_frames
        and abs(source_fps - config.target_fps) <= FPS_TOLERANCE
    )

    if uses_original:
        manifest = {
            "version": MANIFEST_VERSION,
            "source": _source_signature(source_path),
            "target_fps": config.target_fps,
            "target_num_frames": config.target_num_frames,
            "short_video_strategy": config.short_video_strategy,
            "source_fps": source_fps,
            "source_frame_count": source_frame_count,
            "resampled_frame_count": target_frame_count,
            "usable_frame_count": source_frame_count,
            "clip_paths": [str(source_path)],
            "used_original": True,
        }
        _write_manifest(_manifest_path(source_path, config), manifest)
        print(f"Using control video without preprocessing: {source_path.name} ({source_frame_count} frames @ {source_fps:.3f} fps)")
        return [source_path]

    full_clip_count = target_frame_count // config.target_num_frames
    pad_to_full_clip = False
    usable_frame_count = full_clip_count * config.target_num_frames

    if full_clip_count == 0:
        if config.short_video_strategy == "error":
            raise ValueError(
                f"Control video {source_path} becomes {target_frame_count} frames at {config.target_fps} fps, "
                f"which is shorter than the required {config.target_num_frames} frames."
            )
        pad_to_full_clip = True
        clip_count = 1
        usable_frame_count = target_frame_count
    else:
        clip_count = full_clip_count

    sample_indices = _build_sample_indices(
        source_frame_count=source_frame_count,
        source_fps=source_fps,
        target_fps=config.target_fps,
        output_frame_count=usable_frame_count,
    )
    clip_paths = _write_clip_frames(
        source_path=source_path,
        source_frame_count=source_frame_count,
        artifact_dir=artifact_dir,
        sample_indices=sample_indices,
        config=config,
        clip_count=clip_count,
        pad_to_full_clip=pad_to_full_clip,
    )

    manifest = {
        "version": MANIFEST_VERSION,
        "source": _source_signature(source_path),
        "target_fps": config.target_fps,
        "target_num_frames": config.target_num_frames,
        "short_video_strategy": config.short_video_strategy,
        "source_fps": source_fps,
        "source_frame_count": source_frame_count,
        "resampled_frame_count": target_frame_count,
        "usable_frame_count": usable_frame_count,
        "clip_paths": [str(path) for path in clip_paths],
        "used_original": False,
        "padded_short_video": pad_to_full_clip,
        "dropped_resampled_frames": max(0, target_frame_count - usable_frame_count),
    }
    _write_manifest(_manifest_path(source_path, config), manifest)

    print(
        "Prepared control video "
        f"{source_path.name}: {source_frame_count} frames @ {source_fps:.3f} fps -> "
        f"{target_frame_count} frames @ {config.target_fps} fps -> {len(clip_paths)} clip(s)"
    )
    return clip_paths


def prepare_control_videos(
    source_paths: List[Path],
    config: ControlVideoPrepConfig,
    rank: int = 0,
) -> List[Path]:
    source_paths = [path.resolve() for path in source_paths]

    if rank == 0:
        prepared = []
        for source_path in source_paths:
            prepared.extend(prepare_control_video(source_path, config=config, allow_create=True))
        if not prepared:
            raise ValueError("No usable control clips were prepared.")
        return prepared

    _wait_for_preprocessed_manifests(source_paths, config=config)
    prepared = []
    for source_path in source_paths:
        prepared.extend(prepare_control_video(source_path, config=config, allow_create=False))

    if not prepared:
        raise ValueError("No usable control clips were prepared.")
    return prepared
