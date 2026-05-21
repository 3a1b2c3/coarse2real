import argparse
import json
import subprocess
import time
from contextlib import nullcontext
from pathlib import Path

import imageio_ffmpeg
import torch
import torch.distributed as dist
from tqdm import tqdm

from c2r import VideoData, save_video

try:
    from inference.inference_utils import (
        build_pipeline,
        build_tasks,
        load_config,
        maybe_init_dist,
        prepare_control_videos,
        read_prompts,
    )
except ModuleNotFoundError:
    from inference_utils import (  # type: ignore
        build_pipeline,
        build_tasks,
        load_config,
        maybe_init_dist,
        prepare_control_videos,
        read_prompts,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WAN + DINO inference from a JSON config.")
    parser.add_argument("config", nargs="?", default=None, help="Path to inference JSON config file.")
    parser.add_argument("--config", dest="config_opt", default=None, help="Path to inference JSON config file.")
    args = parser.parse_args()
    config_path = args.config_opt or args.config
    if config_path is None:
        parser.error("config path is required (use positional `config` or `--config`).")
    return argparse.Namespace(config=config_path)


def barrier_on_local_cuda_device() -> None:
    if not dist.is_initialized():
        return
    dist.barrier()


def make_progress_bar(rank: int, mode: str):
    if mode == "usp" and rank != 0:
        return lambda iterable: iterable
    return tqdm


def local_task_prompt_subset(tasks: list[dict], rank: int, world_size: int, mode: str) -> list[str]:
    if mode != "dp" or world_size <= 1:
        return list(dict.fromkeys(task["prompt"] for task in tasks))

    task_prompts = [
        task["prompt"]
        for task in tasks
        if task["task_id"] % world_size == rank
    ]
    return list(dict.fromkeys(task_prompts))


def wait_for_prompt_enhancement_artifacts(out_dir: Path, rank: int, timeout_seconds: float = 1800.0) -> None:
    if rank == 0:
        return

    artifacts_dir = out_dir / "prompt_enhancement"
    expected_paths = [
        artifacts_dir / "video_descriptions.json",
        artifacts_dir / "enhanced_prompts.jsonl",
    ]
    deadline = time.monotonic() + timeout_seconds
    while True:
        if all(path.is_file() for path in expected_paths):
            return
        if time.monotonic() >= deadline:
            missing = ", ".join(str(path) for path in expected_paths if not path.is_file())
            raise TimeoutError(
                "Timed out waiting for rank 0 to write prompt enhancement artifacts. "
                f"Missing: {missing}"
            )
        time.sleep(1.0)


def vstack_with_source(generated_path: Path, source_path: Path, out_path: Path) -> None:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    # Preprocessed control clips are saved at their native (source) resolution, which differs
    # from the diffusion output dimensions. scale2ref rescales the source to match the
    # generated clip so vstack's identical-width requirement is satisfied.
    filter_complex = (
        "[1:v][0:v]scale2ref=flags=lanczos[src][gen];"
        "[gen][src]vstack=inputs=2"
    )
    cmd = [
        ffmpeg,
        "-y",
        "-loglevel", "error",
        "-i", str(generated_path),
        "-i", str(source_path),
        "-filter_complex", filter_complex,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)


def maybe_prime_prompt_cache(
    pipe,
    prompts: list[str],
    negative_prompt: str,
    prompt_cache_autoprime_threshold: int | None,
    rank: int,
) -> None:
    unique_prompt_count = len(dict.fromkeys(prompts))
    if pipe._prompt_embedding_cache_limit <= 0:
        if rank == 0:
            print("Prompt embedding cache disabled (prompt_embedding_cache_limit=0); skipping startup prompt priming.")
        return

    if prompt_cache_autoprime_threshold == 0:
        if rank == 0:
            print("Prompt cache autopriming disabled (prompt_cache_autoprime_threshold=0); prompts will be cached lazily.")
        return

    if prompt_cache_autoprime_threshold is not None and unique_prompt_count > prompt_cache_autoprime_threshold:
        if rank == 0:
            print(
                "Skipping startup prompt priming because the prompt count exceeds the autoprime threshold "
                f"({unique_prompt_count} > {prompt_cache_autoprime_threshold})."
            )
        return

    primed_count = pipe.prime_prompt_cache(prompts, negative_prompt)
    if rank == 0:
        print(
            f"Pre-primed prompt cache with {primed_count} prompt(s) "
            f"(unique prompts this rank={unique_prompt_count}, cache_limit={pipe._prompt_embedding_cache_limit})."
        )


def run(config_path: str) -> None:
    cfg = load_config(config_path)
    prompt_enhancement_mode = cfg.normalized_prompt_enhancement_mode()

    mode = cfg.normalized_parallel_mode()
    rank, world_size, local_rank = maybe_init_dist(mode)
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = read_prompts(cfg.prompts_file)
    control_video_paths = prepare_control_videos(cfg, rank=rank)

    if prompt_enhancement_mode == "enhanced":
        from c2r.prompt_enhancement import load_prompt_enhancement_artifacts, prepare_prompt_enhancement

        enhancement_dtype = (
            (torch.bfloat16 if cfg.dtype == "bfloat16" else torch.float16)
            if torch.cuda.is_available()
            else torch.float32
        )
        if rank == 0:
            vlm_name = cfg.prompt_enhancement_vlm_model_id.split("/")[-1]
            llm_name = cfg.prompt_enhancement_llm_model_id.split("/")[-1]
            print(
                f"Preparing enhanced prompts with {vlm_name} + {llm_name} "
                f"using {cfg.prompt_enhancement_num_frames} frame(s) for layout/subject motion "
                f"and {cfg.prompt_enhancement_camera_num_frames} frame(s) for camera motion per control clip, "
                f"followed by text cleanup and prompt fusion..."
            )
            prepare_prompt_enhancement(
                prompts=prompts,
                control_video_paths=control_video_paths,
                output_dir=out_dir,
                cache_dir=cfg.prompt_enhancement_cache_dir,
                height=cfg.height,
                width=cfg.width,
                sample_frames=cfg.prompt_enhancement_num_frames,
                camera_sample_frames=cfg.prompt_enhancement_camera_num_frames,
                vlm_model_id=cfg.prompt_enhancement_vlm_model_id,
                llm_model_id=cfg.prompt_enhancement_llm_model_id,
                device=f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu",
                torch_dtype=enhancement_dtype,
                rank=rank,
            )
            print(f"Prompt enhancement preprocessing complete. Artifacts saved under {out_dir / 'prompt_enhancement'}.")
        if dist.is_initialized():
            dist.barrier()
        else:
            wait_for_prompt_enhancement_artifacts(out_dir, rank=rank)
        enhancement_artifacts = load_prompt_enhancement_artifacts(out_dir)
        tasks = build_tasks(
            prompts,
            control_video_paths,
            prompt_overrides=enhancement_artifacts.prompt_overrides,
            control_video_descriptions=enhancement_artifacts.control_video_descriptions,
        )
    else:
        tasks = build_tasks(prompts, control_video_paths)

    pipe = build_pipeline(cfg, local_rank=local_rank, rank=rank)
    maybe_prime_prompt_cache(
        pipe,
        local_task_prompt_subset(tasks, rank=rank, world_size=world_size, mode=mode),
        cfg.negative_prompt,
        cfg.prompt_cache_autoprime_threshold,
        rank,
    )
    if rank == 0:
        print(f"Loaded prompts: {len(prompts)} from {cfg.prompts_file}")
        print(
            f"Prepared control video clips: {len(control_video_paths)} "
            f"(control_videos_dir={cfg.control_videos_dir})"
        )
        print(f"Total generations (prompt x control video): {len(tasks)}")

    metadata_path = out_dir / f"results_rank{rank}.jsonl"
    progress_bar_cmd = make_progress_bar(rank, mode)
    control_video_cache: dict[str, VideoData] = {}
    metadata_context = (
        metadata_path.open("a", encoding="utf-8")
        if mode != "usp" or rank == 0
        else nullcontext(None)
    )

    with metadata_context as meta_file:
        for task in tasks:
            if mode == "dp" and world_size > 1 and (task["task_id"] % world_size != rank):
                continue

            control_video_path = str(task["control_video_path"])
            control_video = control_video_cache.get(control_video_path)
            if control_video is None:
                control_video = VideoData(control_video_path, height=cfg.height, width=cfg.width)
                control_video_cache[control_video_path] = control_video
            render_seed = cfg.seed
            video = pipe(
                prompt=task["prompt"],
                negative_prompt=cfg.negative_prompt,
                control_video=control_video,
                height=cfg.height,
                width=cfg.width,
                num_frames=cfg.num_frames,
                num_inference_steps=cfg.steps,
                seed=render_seed,
                guidance_mode=cfg.guidance_mode,
                control_video_scale=cfg.control_video_scale,
                text_scale=cfg.text_scale,
                apg_eta=cfg.apg_eta,
                apg_momentum=cfg.apg_momentum,
                apg_norm_threshold=cfg.apg_norm_threshold,
                apg_eps=cfg.apg_eps,
                apg_eta_control_video=cfg.apg_eta_control_video,
                apg_eta_text=cfg.apg_eta_text,
                apg_momentum_control_video=cfg.apg_momentum_control_video,
                apg_momentum_text=cfg.apg_momentum_text,
                apg_norm_threshold_control_video=cfg.apg_norm_threshold_control_video,
                apg_norm_threshold_text=cfg.apg_norm_threshold_text,
                control_video_guidance_end=cfg.control_video_guidance_end,
                vae_decode_mode=cfg.normalized_vae_decode_mode(),
                progress_bar_cmd=progress_bar_cmd,
            )
            if mode == "usp" and rank != 0:
                continue

            file_name = f"p{task['prompt_id']:04d}_c{task['control_video_id']:04d}_{Path(task['control_video_path']).stem}.mp4"
            out_path = out_dir / file_name
            save_video(video, str(out_path), fps=cfg.fps, video_encoding_quality=cfg.video_encoding_quality)

            joined_path = out_path.with_name(f"{out_path.stem}_with_source.mp4")
            vstack_with_source(out_path, Path(task["control_video_path"]), joined_path)

            record = {**task, "seed": render_seed, "output": str(out_path), "joined": str(joined_path)}
            if meta_file is not None:
                meta_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                meta_file.flush()

    if dist.is_initialized():
        # DP ranks are independent task workers, so a final NCCL barrier is unnecessary and
        # can hang if NCCL has to guess the wrong device. Keep the synchronization only for USP.
        if mode == "usp":
            barrier_on_local_cuda_device()
        dist.destroy_process_group()


if __name__ == "__main__":
    args = parse_args()
    run(args.config)
