#!/usr/bin/env bash
# Judge the ls_text arms as they land, without ever putting two clients on one
# NHR@FAU deployment.
#
# The free3 panel is Qwen3.6, DeepSeek and gpt-oss. Two of those are also
# generators in run_ls_text_nods.sh, and DeepSeek is generating the oracle
# top-10 arm in another process. run_oracle_top10.sh measured what the overlap
# costs: 43 minutes for zero essays, with judges falling from ~30s to ~300s per
# essay, because three clients on one model is slower in total than the same
# work serialised.
#
# So judging waits for a window in which no seat model is busy:
#
#   * no `oracle_experiment.py run` anywhere  -> the DeepSeek seat is free
#   * the current ls_text generator is not Qwen3.6 and not gpt-oss
#
# Generation order makes those windows long: Qwen3.6 and gpt-oss are first and
# third, so once generation reaches Qwen3-Next everything after it is a
# non-seat model and judging can run uninterrupted.
#
# Idempotent: judge() tops up only the seats a file lacks, so an arm judged
# during a short window and interrupted is simply finished on the next pass.
#
#     setsid nohup bash analysis/run_ls_text_judge_watch.sh > logs/ls_text_judge.log 2>&1 < /dev/null &
set -u
P=~/miniconda3/envs/plexam/bin/python
export PYTHONPATH=analysis
D=zubaers_result/essay_writing/rag_titled_combined_ls_text/ji2

MODELS="Qwen/Qwen3.6-35B-A3B-FP8 RedHatAI/gemma-4-31B-it-FP8-block \
        openai/gpt-oss-120b qwen3-next-80b-a3b-instruct \
        GaleneAI/Magistral-Small-2509-FP8-Dynamic \
        RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-FP8"

#: which generator is busy right now, as an endpoint id, or empty
busy_generator() {
  pgrep -af 'corpus_rag_run\.py generate' 2>/dev/null \
    | grep -oE '\-\-models [^ ]+' | awk '{print $2}' | head -1
}

#: true when no seat model is occupied by a generation
window_open() {
  pgrep -f 'oracle_experiment\.py run' > /dev/null && return 1   # DeepSeek seat
  local g; g=$(busy_generator)
  case "$g" in
    *Qwen3.6*|*gpt-oss*) return 1 ;;                             # Qwen3.6/gpt-oss seats
    *) return 0 ;;
  esac
}

#: arms whose CSV exists but is missing at least one free3 seat
unjudged() {
  $P - <<'PY' 2>/dev/null
import glob, os
import pandas as pd
SEATS = ['score_Judge (Qwen3.6-35B-A3B-FP8)',
         'score_Judge (DeepSeek-V4-Flash)',
         'score_Judge (gpt-oss-120b)']
D = 'zubaers_result/essay_writing/rag_titled_combined_ls_text/ji2'
for p in sorted(glob.glob(f'{D}/rag_*.csv')):
    try:
        cols = pd.read_csv(p, nrows=0).columns
    except Exception:
        continue
    if any(s not in cols for s in SEATS):
        print(os.path.basename(p)[4:-4])
PY
}

slug_to_model() {  # file slug -> endpoint id
  local s=$1 m
  for m in $MODELS; do [ "${m//\//_}" = "$s" ] && { echo "$m"; return; }; done
}

idle=0
while [ $idle -lt 240 ]; do
  todo=$(unjudged)
  gen_running=$(pgrep -f 'run_ls_text_nods\.sh' > /dev/null && echo yes || echo no)
  if [ -z "$todo" ]; then
    # "no work and no generator" is also what the gap between two generation
    # runs looks like, so require an arm on disk before believing it.
    if [ "$gen_running" = no ] && [ -n "$(ls $D/rag_*.csv 2>/dev/null)" ]; then
      echo "### $(date '+%F %T')  nothing unjudged and generation finished"; break
    fi
    idle=$((idle+1)); sleep 120; continue
  fi
  if ! window_open; then
    echo "### $(date '+%F %T')  seat busy ($(busy_generator)${_:-}), waiting"
    sleep 180; continue
  fi
  idle=0
  for s in $todo; do
    window_open || { echo "### $(date '+%F %T')  window closed mid-pass"; break; }
    m=$(slug_to_model "$s")
    [ -z "$m" ] && { echo "### unknown slug $s"; continue; }
    echo "### $(date '+%F %T')  judge ls_text $m"
    $P -u analysis/corpus_rag_run.py judge --pipeline ls_text --top-k 10 \
       --corpora combined --judges free3 --yes --models "$m" \
      || echo "### $(date '+%F %T')  FAILED judge $m"
  done
done
echo "### $(date '+%F %T')  ls_text judging watcher done"
