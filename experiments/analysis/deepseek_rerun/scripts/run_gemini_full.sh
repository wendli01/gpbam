#!/usr/bin/env bash
# Full gemini-3.7-flash run: 81 no-RAG essays, 200 recitation items, then judging
# by the same three FAU seats the rest of the table uses.
#
# Piloted first: 10 items, all finish_reason=stop, $3.70 projected for the whole
# thing. Two workers, not three -- three concurrent requests to Google through
# OpenRouter returned upstream errors on 8 of 10 calls.
#
# Every stage is resumable at item granularity, so re-running this script after an
# interruption picks up where it stopped rather than re-billing what already landed.
set -u
source ~/miniconda3/etc/profile.d/conda.sh && conda activate plexam
cd "$(git rev-parse --show-toplevel)"
S=experiments/analysis/deepseek_rerun/scripts
M=google/gemini-3.7-flash
SLUG=gemini-3.7-flash
GEN=experiments/zubaers_result/essay_writing/frontier/${SLUG}_essays.jsonl

echo "### $(date -Is)  generation: essays + recitation"
RUN_MODEL=$M RUN_SLUG=$SLUG RUN_ENDPOINT=openrouter RUN_EFFORT=high RUN_WORKERS=2 \
  RUN_TASKS=essay,recitation python -u $S/run_model.py || echo '### generation exited non-zero'

echo "### $(date -Is)  judging on the three FAU seats"
JUDGE_GEN=$GEN JUDGE_WAIT=0 \
  JUDGE_OUTD=experiments/zubaers_result/essay_writing/frontier/judged_${SLUG} \
  python -u $S/judge_0731.py || echo '### judging exited non-zero'

echo "### $(date -Is)  DONE"
