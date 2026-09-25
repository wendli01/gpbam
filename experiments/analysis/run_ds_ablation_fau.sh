#!/usr/bin/env bash
# The DeepSeek oracle ablation on NHR@FAU, after OpenRouter produced unusable arms.
#
# What happened on OpenRouter. `deepseek/deepseek-v4-flash-0731` there refuses
# the task as the oracle context grows: 18 of 40 essays in _gold_combined
# declared the case facts missing and wrote no Gutachten, against 1 of 81 in
# _gold_top10. Paired on the same 40 cases that is -23.00 +- 3.99, the wrong
# sign -- more oracle context cannot score worse than less. The prompts were
# verified clean first: the complete Sachverhalt sits verbatim in the Frage
# fence in 5/5 sampled cases, and the norm contexts hold no ellipses and no
# empty texts. The published ablation ran this same prompt at 35.0 norms/case
# on FAU and returned +4.13, so the defect is the provider's serving of the
# weights, not this repo's prompt construction.
#
# Hence back to FAU, at its default concurrency of 10. Raising it is what took
# DeepSeek from ~49 to ~10 tok/s per stream earlier today, so --concurrency is
# deliberately not passed here.
#
# Cost: free. Speed: ~20 min/essay at 5-10 concurrent, so ~5h per arm. Eight
# arms does not fit before submission; the order below front-loads the two the
# draft actually needs -- the anchor, then the top-10 cell that is currently
# printed as "--" in tab:oracle-ablation.
#
#     setsid nohup bash analysis/run_ds_ablation_fau.sh > logs/ds_ablation_fau.log 2>&1 < /dev/null &
set -u
P=~/miniconda3/envs/plexam/bin/python
export PYTHONPATH=analysis
M=deepseek-ai/DeepSeek-V4-Flash          # endpoints.yaml aliases this to -0731
SLUG=${M//\//_}
D=zubaers_result/essay_writing/oracle_rag/ji2
J=Qwen/Qwen3.6-35B-A3B-FP8
PAD_FROM=zubaers_result/essay_writing/rag_titled_combined_rrf_k50/ji2/rag_${SLUG}.csv

wait_up() {
  local n=0
  until $P analysis/endpoint_probe.py "$M" 1500 180 2>&1 | grep -q 'up,'; do
    n=$((n+1)); [ $((n % 6)) -eq 1 ] && echo "### $(date '+%F %T')  $M down, waiting"
    sleep 300
  done
}

orc() {  # tag, extra args...
  local tag=$1; shift
  [ -f "$D/oracle_${SLUG}${tag}.csv" ] && {
    echo "### $(date '+%F %T')  $tag exists, skipping"; return 0; }
  local n
  for n in $(seq 1 12); do
    echo "### $(date '+%F %T')  $tag attempt $n/12"
    wait_up
    if timeout 6h $P -u analysis/oracle_experiment.py run --model "$M" \
         --tag "$tag" --judge "$J" --max-passages "${K:-120}" --chunk 10; then
      return 0
    fi
    echo "### $(date '+%F %T')  $tag attempt $n exited $?"
    [ -f "$D/oracle_${SLUG}${tag}.csv.part" ] && \
      echo "###   banked $(( $(wc -l < "$D/oracle_${SLUG}${tag}.csv.part") - 1 )) row(s)"
    sleep 120
  done
  echo "### $(date '+%F %T')  $tag FAILED after 12 attempts"; return 1
}

orc _gold_combined   || exit 1      # the anchor; nothing prints without it
K=10 orc _gold_top10 || exit 1      # the cell the draft shows as "--"
orc _gold_now        || exit 1
orc _gold_unc  --include-uncovered
orc _gold_shuf --shuffle-refs
if [ ! -f "$PAD_FROM" ]; then
  $P -u analysis/corpus_rag_run.py generate --pipeline rrf --top-k 50 \
     --corpora combined --models "$M" || echo "### FAILED pad-source"
fi
[ -f "$PAD_FROM" ] && orc _gold_pad --shuffle-refs --pad-junk 20 --pad-from "$PAD_FROM"
orc _gold_cites --cites-only --cites-from "$D/oracle_${SLUG}_gold_now.csv"

echo "### $(date '+%F %T')  filling the other two free3 seats"
$P -u analysis/oracle_experiment.py rejudge --judges free3 --arm gold \
  || echo '### rejudge exited non-zero'
echo "### $(date '+%F %T')  done"
