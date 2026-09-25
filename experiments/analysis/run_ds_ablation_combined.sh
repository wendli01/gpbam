#!/usr/bin/env bash
# DeepSeek's oracle ablation, regenerated against the combined corpus.
#
# Why: its five conditions are the 08-20 arms, generated when
# oracle_experiment.KB still pointed at the federal knowledge base -- 35.0 of
# the 51.9 norms the solution cites per case and 34.4% of the state-law ones,
# against 47.9 and 97.4% for Gemma-4-31B and gpt-oss, which were re-run on
# 09-09. The oracle column of tab:oracle-ablation now reads _gold_combined,
# the same arm the ladder reads, so a DeepSeek row would anchor on the
# combined corpus while the five conditions it anchors sat on the federal one.
# The row is out of the table until this lands.
#
# _gold_now is regenerated too even though it is no longer printed: _gold_cites
# lifts its norm list verbatim, so the leakage test names whatever selection
# that file holds. A federal _gold_now would put a federal list in a combined
# arm.
#
# _gold_top10 never existed for DeepSeek. run_ds_top10.sh tried seventeen times
# across 09-10 and 09-11 and wrote nothing: GEN_CHUNK was 20 and the attempt cap
# 45 minutes, so at the ~10 min/case the endpoint was serving it needed 3h20 to
# reach the first checkpoint and never got there. CHUNK is 5 here.
#
# Serialised behind the LS judging chain on purpose -- it holds a DeepSeek judge
# seat, and run_oracle_top10.sh records what three clients on one model costs:
# 43 minutes for zero essays, judges falling from ~30s to ~300s per essay.
#
#     setsid nohup bash analysis/run_ds_ablation_combined.sh \
#         > logs/ds_ablation_combined.log 2>&1 < /dev/null &
set -u
P=~/miniconda3/envs/plexam/bin/python
export PYTHONPATH=analysis
M=deepseek-ai/DeepSeek-V4-Flash
SLUG=${M//\//_}
D=zubaers_result/essay_writing/oracle_rag/ji2
J=Qwen/Qwen3.6-35B-A3B-FP8

# Wait on the work, not on a pid. `pgrep -f run_ls_judging.sh` also matches the
# interactive shell that launched it, and that wrapper's lifetime has nothing to
# do with the chain's -- it can outlive it and block here forever, or exit first
# and let generation start on top of a running DeepSeek judge seat. Matching the
# judge invocation itself is right either way.
# Anchored, because the shell that launched the chain carries the whole script
# in its own cmdline as a heredoc and matches any substring of it.
BUSY='^(bash analysis/run_ls_judging\.sh|[^ ]*python -u analysis/corpus_rag_run\.py judge)'
while pgrep -f "$BUSY" > /dev/null; do
  echo "### $(date '+%F %T')  judging still running, waiting"
  sleep "${WAIT_EVERY:-300}"
done
echo "### $(date '+%F %T')  no other client on the endpoint"

# ...which is not the same as the endpoint being up. At 11:5x on 09-11 the
# DeepSeek deployment answered four probes with an immediate 500 -- LiteLLM
# could not reach aquavan2.rrze.uni-erlangen.de:8000 at all. A 500 comes back
# in under two seconds, so a 20-attempt retry loop against a dead backend
# exhausts itself in about an hour and writes nothing; that is what happened to
# run_ds_top10.sh. Wait for a real completion before generating anything.
echo "### $(date '+%F %T')  waiting for $M to answer"
until $P analysis/endpoint_probe.py "$M"; do
  sleep "${PROBE_EVERY:-600}"
done
echo "### $(date '+%F %T')  endpoint up, starting"

# The federal arms are evidence, not garbage: the paper's earlier ablation was
# computed off them. Moved aside, not deleted, so the driver stops skipping.
mkdir -p "$D/federal_kb_0820"
for T in _gold_now _gold_unc _gold_shuf _gold_pad _gold_cites; do
  F="$D/oracle_${SLUG}${T}.csv"
  [ -f "$F" ] && { mv -v "$F" "$D/federal_kb_0820/"; }
done

echo "### $(date '+%F %T')  five conditions"
MODEL=$M CHUNK=5 LIMIT=90m ATTEMPTS=20 BACKOFF=180 \
  bash analysis/run_oracle_ablation_model.sh \
  || echo "### $(date '+%F %T')  ablation driver exited non-zero"

echo "### $(date '+%F %T')  _gold_top10"
for n in $(seq 1 20); do
  [ -f "$D/oracle_${SLUG}_gold_top10.csv" ] && break
  echo "### $(date '+%F %T')  top10 attempt $n/20"
  timeout 90m $P -u analysis/oracle_experiment.py run --model "$M" \
     --tag _gold_top10 --judge "$J" --max-passages 10 --chunk 5 && break
  echo "### $(date '+%F %T')  top10 attempt $n exited $?"
  [ -f "$D/oracle_${SLUG}_gold_top10.csv.part" ] \
    && echo "###   banked $(( $(wc -l < "$D/oracle_${SLUG}_gold_top10.csv.part") - 1 )) case(s)"
  sleep 180
done

echo "### $(date '+%F %T')  third free seat"
$P -u analysis/oracle_experiment.py rejudge --judges free3 --arm gold \
  || echo '### rejudge exited non-zero'

echo "### $(date '+%F %T')  done. Put DeepSeek back in GEN in"
echo "###   analysis/oracle_ablation_table.py and regenerate."
