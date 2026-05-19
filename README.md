<h1 align="center">C2R: Coarse-to-Real</h1>

<p align="center">
  <a href="https://gonzalogn.com/">Gonzalo Gomez-Nogales</a><sup>1</sup>,
  <a href="https://yiconghong.me/">Yicong Hong</a><sup>2</sup>,
  <a href="https://chongjiange.github.io/">Chongjian Ge</a><sup>2</sup>,
  Peiye Zhuang<sup>3</sup>,
  <a href="https://dancasas.github.io/">Dan Casas</a><sup>1</sup>,
  <a href="https://zhouyisjtu.github.io/">Yi Zhou</a><sup>3</sup>
</p>

<p align="center">
  <sup>1</sup>Universidad Rey Juan Carlos&nbsp;&nbsp;
  <sup>2</sup>Adobe Research&nbsp;&nbsp;
  <sup>3</sup>Roblox
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2601.22301">
    <img src="https://img.shields.io/badge/arXiv-Paper-b31b1b.svg" alt="arXiv Paper">
  </a>
  <a href="https://huggingface.co/gonsaBRK/coarse2real">
    <img src="https://img.shields.io/badge/🤗%20Model-Weights-yellow" alt="Model Weights">
  </a>
</p>

<p align="justify">
  Traditional rendering pipelines rely on complex assets, accurate materials and lighting, and substantial computational resources to produce realistic imagery, yet they still face challenges in scalability and realism for populated dynamic scenes. We present C2R (Coarse-to-Real), a generative rendering framework that synthesizes real-style urban crowd videos from coarse 3D simulations. Our approach uses coarse 3D renderings to explicitly control scene layout, camera motion, and human trajectories, while a learned neural renderer generates realistic appearance, lighting, and fine-scale dynamics guided by text prompts. To overcome the lack of paired training data between coarse simulations and real videos, we adopt a two-stage synthetic-real domain-hedging strategy that first learns a strong generative prior from large-scale real footage, and then introduces controllability by using a small amount of paired synthetic coarse-to-fine data to anchor shared implicit spatio-temporal features across domains. The resulting system supports coarse-to-fine control, generalizes across diverse CG and game inputs, and produces temporally consistent, controllable, and realistic urban scene videos from minimal 3D input.
</p>

This repository contains the inference-only release for **C2R (Coarse-to-Real)**, a generative rendering model that turns coarse 3D simulation videos into realistic controlled videos using a prompt and the 3D coarse simulation video as input.

The release path is:

- 14B C2R inference only
- One or more control videos are always required
- The C2R DINO adapter checkpoint is always required
- Supported execution modes are single GPU, USP multi-GPU for one result, and DP multi-GPU for many results

## Installation

```bash
git clone https://github.com/GonzaloGNogales/coarse2real.git
cd coarse2real

conda env create -f c2r-setup.yml
conda activate coarse2real
```

The default environment is the recommended path for the release and includes Python 3.11, PyTorch 2.8.0, CUDA 12.8, `flash-attn-4`, and the pinned runtime dependencies.

No extra install step is needed after creating the environment. If you run from the repo root, `python -m inference.run_inference ...` it will work directly.

## Weights

Download the **Wan2.1 14B** base weights under:

```text
models/wan/
```

Use the official Wan2.1 14B Hugging Face repository:

```bash
mkdir -p models/wan
hf download Wan-AI/Wan2.1-T2V-14B \
  --local-dir models/wan
```

We release our C2R DiT checkpoints separately, but the codebase still expects the Wan2.1 14B base folder to provide the text encoder, VAE, and tokenizer assets. Expected files include:

```text
models/wan/models_t5_umt5-xxl-enc-bf16.pth
models/wan/Wan2.1_VAE.pth
models/wan/google/umt5-xxl/...
```

Once access to model weights is granted, download the released C2R 14B backbone from the C2R Hugging Face repository:

```bash
mkdir -p models/pretrained_dit_backbone
hf download gonsaBRK/coarse2real c2r-dit-backbone-14B.safetensors \
  --local-dir models/pretrained_dit_backbone
```

Download the released DINO adapter from the same repository:

```bash
mkdir -p models/dino_adapter
hf download gonsaBRK/coarse2real c2r-dino-adapter.safetensors \
  --local-dir models/dino_adapter
```

C2R uses the DINOv3 backbone `facebook/dinov3-vitb16-pretrain-lvd1689m` for control-video features. For offline or cluster runs, download it once and point the configs at the local folder:

```bash
mkdir -p models/dino/dinov3-vitb16-pretrain-lvd1689m
hf download facebook/dinov3-vitb16-pretrain-lvd1689m \
  --local-dir models/dino/dinov3-vitb16-pretrain-lvd1689m
```

Then set:

```json
"dino_model_path": "models/dino/dinov3-vitb16-pretrain-lvd1689m"
```

If `dino_model_path` is `null`, rank 0 uses the Hugging Face cache and may download the model if network access is available.

## Inputs

Prompts are read from the config `prompts_file`, one prompt per line.

The shipped configs read prompts from:

```text
inference/c2r-prompts.txt
```

Control videos are read recursively from `control_videos_dir`. The shipped configs use:

```text
inference/control_videos
```

Add your own coarse videos there, or replace the folder path in the config. Supported extensions are:

```text
.mp4 .mov .mkv .avi .webm .m4v
```

By default, control videos are preprocessed to the configured frame count and FPS before inference.

## Configs

Only these release configs are shipped:

```text
inference/config_1gpu.json
inference/config_multigpu_usp.json
inference/config_multigpu_dp.json
```

Important fields:

- `base_model_dir`: local WAN 14B folder
- `dit_path`: released C2R 14B backbone
- `dino_adapter_path`: released C2R DINO adapter
- `dino_model_path`: optional local DINOv3 folder
- `prompts_file`: prompt list
- `control_videos_dir`: control-video folder
- `parallel_mode`: `single-gpu`, `usp`, or `dp`
- `prompt_enhancement_mode`: `off` or `enhanced`

`control_videos_dir` and `dino_adapter_path` are required.

## Run Inference

Single GPU:

```bash
bash inference/launch_1gpu.sh
```

USP multi-GPU, one generated result at a time using multiple GPUs:

```bash
bash inference/launch_multigpu_usp.sh
```

DP multi-GPU, many generated results at the same time with one worker per GPU:

```bash
bash inference/launch_multigpu_dp.sh
```

The multi-GPU launchers infer `NPROC_PER_NODE` from `CUDA_VISIBLE_DEVICES`, scheduler variables, PyTorch CUDA visibility, or `nvidia-smi`. You can override it explicitly:

```bash
NPROC_PER_NODE=8 bash inference/launch_multigpu_usp.sh
NPROC_PER_NODE=8 bash inference/launch_multigpu_dp.sh
```

You can also run a config directly:

```bash
python -m inference.run_inference --config inference/config_1gpu.json
torchrun --standalone --nproc_per_node=8 -m inference.run_inference --config inference/config_multigpu_usp.json
torchrun --standalone --nproc_per_node=8 -m inference.run_inference --config inference/config_multigpu_dp.json
```

## Gradio Demo

Launch the local demo:

```bash
bash inference/launch_gradio.sh
```

By default it binds to `127.0.0.1:7860`, uses `inference/config_multigpu_usp.json` to load default config parameters (can be changed through the UI), reads example videos from `inference/control_videos`, and exposes C2R workflow: choose or upload one control video, write one prompt, optionally enable prompt enhancement, and run with either single GPU or USP.

If you launch the app on a remote cluster, `127.0.0.1` is local to the cluster node. Open an SSH tunnel from your own computer, then browse to `http://127.0.0.1:7860` locally:

```bash
ssh -L 7860:127.0.0.1:7860 your_user@cluster-login-host
```

## Prompt Enhancement

Prompt enhancement is optional for C2R runs:

```json
"prompt_enhancement_mode": "enhanced"
```

It uses Qwen3 VLM/LLM models to describe the control video and fuse that information with the prompt before generation. This option should increase the quality of the generations at the cost of some preprocessing time at the begining of the inference. Leave it as `off` for the fastest startup.

## Outputs

Generated videos and per-run metadata are written under the config `output_dir`. CLI runs write `results_rank*.jsonl`; the Gradio demo also writes a session manifest with the resolved config, selected control video, prompt, logs, and generated files.

## Notes

- The automatic attention backend prefers the fastest installed backend supported by the current GPU and falls back to PyTorch SDPA when no other option is available.
- The USP launcher runs a NCCL preflight and caches the selected transport mode for the host/GPU visibility set.
- DP is for throughput across many prompt/control-video pairs; USP is for splitting one generation across multiple GPUs.

## Limitations

This is an inference-only research release. The generated videos may contain visual artifacts, temporal inconsistencies, inaccurate fine details, or deviations from the input prompt. Performance may vary depending on the quality, structure, and domain of the coarse control video.

The model is optimized for coarse 3D simulation videos of populated urban scenes. Results outside this domain may be less reliable.

## License

The inference code in this repository is released under the PolyForm Noncommercial License 1.0.0. You may use, modify, and share the code for non-commercial research and evaluation purposes. See [LICENSE](LICENSE).

The model weights are released separately under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0) license and require gated access through Hugging Face.

Allowed for the model weights:

- sharing the original work with attribution
- using the weights for non-commercial research and education

Not allowed for the model weights:

- commercial use
- redistribution of modified versions

Third-party dependencies and model weights are subject to their own licenses.

## Citation

If you use this work in academic research, use this citation entry:

```bibtex
@misc{gomeznogales2026coarsetoreal,
  title         = {Coarse-to-Real: Generative Rendering for Populated Dynamic Scenes},
  author        = {Gomez-Nogales, Gonzalo and Hong, Yicong and Ge, Chongjian and Zhuang, Peiye and Comino-Trinidad, Marc and Casas, Dan and Zhou, Yi},
  year          = {2026},
  eprint        = {2601.22301},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  doi           = {10.48550/arXiv.2601.22301},
  url           = {https://arxiv.org/abs/2601.22301}
}
```

## Contact

For questions or collaborations, please contact:

* **Name:** Gonzalo Gomez-Nogales
* **Email:** [[gonzalo.gomez@urjc.es](mailto:gonzalo.gomez@urjc.es)]

* **Name:** Yi Zhou
* **Email:** [[yizhou@roblox.com](mailto:yizhou@roblox.com)]
* **Email:** [[zhouyisjtu2012@gmail.com](mailto:zhouyisjtu2012@gmail.com)]

