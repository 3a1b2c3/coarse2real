#!/usr/bin/env bash
set -euo pipefail

python -m inference.run_inference --config inference/config_1gpu.json
