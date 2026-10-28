#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
if [[ "$(uname -s)" != "Linux" ]]; then
  echo 'Run this script on the Oracle Linux ARM instance.' >&2
  exit 1
fi
# Run this on the actual Linux ARM instance. No performance extrapolation.
if [[ ! -x .venv/bin/python ]]; then
  TTS_PYTHON_BIN="${TTS_PYTHON_BIN:-python3.12}"
  "$TTS_PYTHON_BIN" -m venv .venv
fi
.venv/bin/python -c 'import sys; assert sys.version_info >= (3, 12), "Pinned dependencies require Python 3.12 or later"'
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python download.py
if [[ ! -d models/fast-fp32-cached ]]; then
  .venv/bin/python prepare.py --mode fp32-cached --output models/fast-fp32-cached
fi
if [[ ! -d models/fast-denoiser-int8 ]]; then
  .venv/bin/python prepare.py --mode denoiser-int8 --output models/fast-denoiser-int8
fi
# Choose one core from this process's permitted affinity mask.
TTS_CPU_CORE="$(.venv/bin/python -c 'import psutil; print(psutil.Process().cpu_affinity()[0])')"
.venv/bin/python benchmark.py --steps 8 --repeats 5 --affinity "$TTS_CPU_CORE" --output results/oracle_fp32_8.json
.venv/bin/python benchmark.py --steps 8 --repeats 5 --affinity "$TTS_CPU_CORE" --model models/fast-fp32-cached --output results/oracle_fast_fp32_8.json
.venv/bin/python benchmark.py --steps 8 --repeats 5 --affinity "$TTS_CPU_CORE" --model models/fast-denoiser-int8 --output results/oracle_fast_int8_8.json
.venv/bin/python experiments/paired_benchmark.py --affinity "$TTS_CPU_CORE" --repeats 5 --variant models/fast-fp32-cached/onnx --output results/oracle_paired_fp32_8.json
.venv/bin/python experiments/paired_benchmark.py --affinity "$TTS_CPU_CORE" --repeats 5 --variant models/fast-denoiser-int8/onnx --output results/oracle_paired_int8_8.json
