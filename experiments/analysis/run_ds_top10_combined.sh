#!/usr/bin/env bash
# The one cell tab:oracle-ablation is missing: DeepSeek _gold_top10.
#
# Against the *combined* corpus, deliberately. The printed table already mixes
# anchors -- DeepSeek's row is the federal _gold_now family (+3.40), Gemma's and
# gpt-oss's are combined _gold_combined (+5.37, +5.86) -- and with two days to
# submission that stays as it is. Combined is the arm wanted afterwards anyway,
# when the whole row is regenerated, so this is the version that does not have
# to be thrown away. It is one dash filled with a number one column-shift out of
# line with its own row, which is a smaller inconsistency than the one the table
# already carries across rows.
#
# Faithful despite the 09-11 deletion: the oracle does not retrieve, so the
# resampled query rewrite cache never reaches it, and the context still reads
# "47.9 of 51.9 cited norms per case" exactly as the published arms did.
#
# Gated on DeepSeek answering at generation length. A 120-token ping returned in
# 1.4s this afternoon while every 9,800-token essay behind it hit the 600s
# timeout, so the gate asks for 1500 tokens.
#
#     setsid nohup bash analysis/run_ds_top10_combined.sh > logs/ds_top10_combined.log 2>&1 < /dev/null &
set -u
P=~/miniconda3/envs/plexam/bin/python
export PYTHONPATH=analysis
M=deepseek-ai/DeepSeek-V4-Flash      # aliased to -0731 in endpoints.yaml
SLUG=${M//\//_}
D=zubaers_result/essay_writing/oracle_rag/ji2
OUT="$D/oracle_${SLUG}_gold_top10.csv"

for n in $(seq 1 40); do
  [ -f "$OUT" ] && { echo "### $(date '+%F %T')  done: $OUT"; break; }
  until $P analysis/endpoint_probe.py "$M" 1500 180 2>&1 | grep -q 'up,'; do
    echo "### $(date '+%F %T')  DeepSeek not generating, waiting"
    sleep 300
  done
  echo "### $(date '+%F %T')  attempt $n/40"
  timeout 3h $P -u analysis/oracle_experiment.py run --model "$M" \
     --tag _gold_top10 --judge Qwen/Qwen3.6-35B-A3B-FP8 \
     --max-passages 10 --chunk 5 && break
  echo "### $(date '+%F %T')  attempt $n exited $?"
  [ -f "$OUT.part" ] && echo "###   banked $(( $(wc -l < "$OUT.part") - 1 )) case(s)"
  sleep 120
done

echo "### $(date '+%F %T')  filling the other two free seats"
$P -u analysis/oracle_experiment.py rejudge --judges free3 --arm gold \
  || echo '### rejudge exited non-zero'
echo "### $(date '+%F %T')  done"
