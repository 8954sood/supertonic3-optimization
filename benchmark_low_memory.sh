#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
TTS_CPU_CORE="$(.venv/bin/python -c 'import psutil; print(psutil.Process().cpu_affinity()[0])')"
for TTS_MODEL_NAME in fast-full-int8 trial-dedup-int8; do
  nice -n 19 .venv/bin/python memory_guard.py \
    --max-rss-mb 420 --reserve-mb 80 --seconds 300 \
    --stats "results/guard_${TTS_MODEL_NAME}.json" -- \
    .venv/bin/python benchmark.py --model "models/${TTS_MODEL_NAME}" \
    --steps 8 --repeats 3 --cases ko_short en_short --affinity "$TTS_CPU_CORE" \
    --output "results/low_memory_${TTS_MODEL_NAME}_8.json"
done
