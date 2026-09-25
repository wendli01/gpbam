#!/bin/bash
# Two remaining panel seats, sequentially, so total added gateway load stays at one batch.
# SEAT_OUT names the seat: each writes results/<SEAT_OUT>.jsonl, appended per model.
cd ${REJUDGE_DIR:-$(git rev-parse --show-toplevel)/experiments/analysis/deepseek_rerun/results}
source ~/miniconda3/etc/profile.d/conda.sh && conda activate plexam
export SEAT_EFFORT=high SEAT_WORKERS=6
echo "===== SEAT 1/2: Qwen/Qwen3.6-35B-A3B-FP8  $(date -Is)"
SEAT='Qwen/Qwen3.6-35B-A3B-FP8' SEAT_OUT=Qwen3.6-35B python -u rejudge_seat.py judge
echo "===== SEAT 2/2: deepseek-ai/DeepSeek-V4-Flash  $(date -Is)"
SEAT='deepseek-ai/DeepSeek-V4-Flash' SEAT_OUT=DeepSeek-V4-Flash-0731 python -u rejudge_seat.py judge
echo "===== ALL SEATS DONE  $(date -Is)"
