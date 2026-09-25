#!/usr/bin/env bash
# Rebuild every arm a `git clean -fd` removed on 2026-09-11.
#
# What was lost: zubaers_result/essay_writing/oracle_rag/ and every
# rag_titled_* directory -- the inputs to tab:ladder, tab:corpus and
# tab:oracle-ablation. What survived, and why this is regeneration rather than
# reconstruction: both knowledge bases, data/gpbam.json, and MAIN_NORAG
# (without_rag_0731/no_rag_ji2_rejudged.csv), which is the baseline every table
# differences against. Nothing has to be rebuilt from source.
#
# Ordered by what the paper needs soonest, because this is days of endpoint
# time and the order decides which table comes back first:
#
#   1  rrf k=10        RETRIEVAL_BASE -- the column the whole retrieval block
#                      of tab:ladder is measured against. Nothing else in that
#                      table can be read until this exists.
#   2  oracle          the ceiling claim, and the anchor tab:oracle-ablation
#                      needs before any condition means anything
#   3  the rest of PAPER_COLS
#   4  federal arms    tab:corpus
#   5  ablation        tab:oracle-ablation's five conditions
#
# Every phase is idempotent: corpus_rag_run splits generate from judge and tops
# up only missing seats, oracle_experiment skips arms whose CSV exists. Re-run
# this after any interruption and it resumes. PHASES selects a subset.
#
#     setsid nohup bash analysis/run_rebuild_all.sh > logs/rebuild_all.log 2>&1 < /dev/null &
#     PHASES="1 2" bash analysis/run_rebuild_all.sh
set -u
P=~/miniconda3/envs/plexam/bin/python
export PYTHONPATH=analysis
PHASES=${PHASES:-"1 2 3 4 5"}

#: the seven generators of tab:ladder, in row order
M7="deepseek-ai/DeepSeek-V4-Flash \
    RedHatAI/gemma-4-31B-it-FP8-block \
    qwen3-next-80b-a3b-instruct \
    Qwen/Qwen3.6-35B-A3B-FP8 \
    openai/gpt-oss-120b \
    GaleneAI/Magistral-Small-2509-FP8-Dynamic \
    RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-FP8"
#: tab:corpus is the five that ran both corpora
M5="deepseek-ai/DeepSeek-V4-Flash RedHatAI/gemma-4-31B-it-FP8-block \
    qwen3-next-80b-a3b-instruct Qwen/Qwen3.6-35B-A3B-FP8 openai/gpt-oss-120b"

has() { case " $PHASES " in *" $1 "*) return 0;; *) return 1;; esac; }

# Wait for a model's endpoint rather than discovering it is gone one arm at a
# time. The DeepSeek group answers 500 in under two seconds when its backend is
# away, so an unguarded queue burns through its whole list in minutes.
wait_up() {
  local m=$1 n=0
  until $P analysis/endpoint_probe.py "$m" > /dev/null 2>&1; do
    n=$((n+1))
    [ $((n % 6)) -eq 1 ] && echo "### $(date '+%F %T')  $m down, waiting"
    [ $n -ge 144 ] && { echo "### $(date '+%F %T')  $m down 12h, skipping"; return 1; }
    sleep 300
  done
  return 0
}

# generate and judge as two calls, never `run`. corpus_rag_run's `run` path
# calls judge(p) with the default FREE_PANEL -- two seats -- and ignores
# --judges entirely; only the `judge` subcommand honours it. An arm judged by
# two of the three free3 seats reads as unjudged, because paper_table.med
# returns None when any seat of the scale is missing, so the whole rebuild
# would have completed and produced empty tables. Both phases are idempotent:
# generate skips a finished CSV, judge tops up only the seats it lacks.
rag() {  # pipeline, top_k, corpora, model
  echo "### $(date '+%F %T')  $1 k=$2 $3 $4"
  wait_up "$4" || return 1
  $P -u analysis/corpus_rag_run.py generate --pipeline "$1" --top-k "$2" \
     --corpora "$3" --models "$4" \
    || { echo "### $(date '+%F %T')  FAILED generate $1 k=$2 $3 $4"; return 1; }
  $P -u analysis/corpus_rag_run.py judge --pipeline "$1" --top-k "$2" \
     --corpora "$3" --judges free3 --yes --models "$4" \
    || echo "### $(date '+%F %T')  FAILED judge $1 k=$2 $3 $4"
}

if has 1; then
  echo "### ===== phase 1: RRF k=10, combined (the ladder's base column) ====="
  for m in $M7; do rag rrf 10 combined "$m"; done
fi

if has 2; then
  echo "### ===== phase 2: oracle (_gold_combined) ====="
  for m in $M7; do
    SLUG=${m//\//_}
    F="zubaers_result/essay_writing/oracle_rag/ji2/oracle_${SLUG}_gold_combined.csv"
    [ -f "$F" ] && { echo "### $(date '+%F %T')  $SLUG oracle exists"; continue; }
    echo "### $(date '+%F %T')  oracle $m"
    wait_up "$m" || continue
    $P -u analysis/oracle_experiment.py run --model "$m" --tag _gold_combined \
       --judge Qwen/Qwen3.6-35B-A3B-FP8 --max-passages 120 --chunk 5 \
      || echo "### $(date '+%F %T')  FAILED oracle $m"
  done
  $P -u analysis/oracle_experiment.py rejudge --judges free3 --arm gold \
    || echo '### rejudge exited non-zero'
fi

if has 3; then
  echo "### ===== phase 3: the rest of PAPER_COLS ====="
  for m in $M7; do
    rag rerank    10 combined "$m"
    rag cite_only 10 combined "$m"
    rag cite_rrf  10 combined "$m"
    rag rrf       50 combined "$m"
    rag cite_rrf  50 combined "$m"
    rag cite_ls_rrf 50 combined "$m"
  done
fi

if has 4; then
  echo "### ===== phase 4: federal arms (tab:corpus) ====="
  for m in $M5; do rag roundrobin 10 federal "$m"; rag rrf 10 federal "$m"; done
fi

if has 5; then
  echo "### ===== phase 5: ablation conditions ====="
  for m in RedHatAI/gemma-4-31B-it-FP8-block openai/gpt-oss-120b; do
    MODEL=$m CHUNK=5 LIMIT=90m ATTEMPTS=20 bash analysis/run_oracle_ablation_model.sh \
      || echo "### $(date '+%F %T')  FAILED ablation $m"
  done
fi

echo "### $(date '+%F %T')  rebuild queue finished"
