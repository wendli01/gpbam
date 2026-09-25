#!/usr/bin/env bash
# The complete DeepSeek oracle ablation, generated on OpenRouter, judged on FAU.
#
# Why OpenRouter for generation. The oracle arms are long: 27,133 completion
# tokens per essay measured on the first banked chunk (the no-RAG baseline runs
# 10,618, so the context roughly triples what the model writes and thinks), at
# ~20 minutes wall-clock each. NHR@FAU degrades under load -- five concurrent
# essays took DeepSeek from ~49 to ~10 tok/s per stream on 09-11 -- so ten at a
# time is the practical ceiling there and an arm costs ~5h. OpenRouter holds its
# per-stream rate at forty, which is ~1.3h per arm, for $0.065/M in + $0.18/M
# out: ~$0.55 per arm, ~$6 for the set. Generation provider is uniform across
# the whole family, so the table's within-family deltas are unaffected.
#
# Why FAU for judging. The judge is the measuring instrument and free3 is the
# paper's scale; moving it to OpenRouter would cost $0.88/arm AND risk a scale
# shift, since 'qwen/qwen3.6-35b-a3b' there is not necessarily the FP8 build FAU
# serves. This project already has one instance of judge weights moving
# underneath it. FAU judges an arm in ~35 minutes, free.
#
# Arms, in dependency order. `orc` skips any arm whose CSV already exists and
# resumes from `.part`, so this is safe to re-run after an interruption.
#
#   _gold_top10      already 10/81 banked at concurrency 10; finishes first
#   _gold_combined   the anchor -- every other condition is differenced against
#                    it, so nothing in the table prints without this one
#   _gold_now        not printed; _gold_cites lifts its norm list verbatim
#   _gold_unc        + norms the corpus lacks, named but not supplied
#   _gold_shuf       solution-derived prominence order destroyed
#   rrf k=50         NOT an oracle arm: the junk source for _gold_pad. The one
#                    arm here that does not reproduce a published context --
#                    qa.ReWriter samples and the published rewrite cache is a
#                    different draw. Junk is still junk; the condition tests
#                    dilution, not any particular norm.
#   _gold_pad        + 20 junk norms, interleaved; pairs against _gold_shuf
#   _gold_cites      the same norms named, all wording withheld
#
#     setsid nohup bash analysis/run_ds_ablation_openrouter.sh > logs/ds_ablation_or.log 2>&1 < /dev/null &
set -u
P=~/miniconda3/envs/plexam/bin/python
export PYTHONPATH=analysis
M=deepseek/deepseek-v4-flash-0731        # falls through endpoints.yaml to openrouter
SLUG=${M//\//_}
D=zubaers_result/essay_writing/oracle_rag/ji2
J=Qwen/Qwen3.6-35B-A3B-FP8               # FAU, free
C=${C:-40}
PAD_FROM=zubaers_result/essay_writing/rag_titled_combined_rrf_k50/ji2/rag_${SLUG}.csv

orc() {  # tag, extra args...
  local tag=$1; shift
  [ -f "$D/oracle_${SLUG}${tag}.csv" ] && {
    echo "### $(date '+%F %T')  $tag exists, skipping"; return 0; }
  local n
  for n in $(seq 1 8); do
    echo "### $(date '+%F %T')  $tag attempt $n/8 (concurrency $C)"
    if timeout 4h $P -u analysis/oracle_experiment.py run --model "$M" \
         --tag "$tag" --judge "$J" --max-passages "${K:-120}" \
         --chunk "$C" --concurrency "$C" "$@"; then
      return 0
    fi
    echo "### $(date '+%F %T')  $tag attempt $n exited $?"
    [ -f "$D/oracle_${SLUG}${tag}.csv.part" ] && \
      echo "###   banked $(( $(wc -l < "$D/oracle_${SLUG}${tag}.csv.part") - 1 )) row(s)"
    sleep 60
  done
  echo "### $(date '+%F %T')  $tag FAILED after 8 attempts"; return 1
}

K=10 orc _gold_top10 || exit 1
orc _gold_combined   || exit 1
orc _gold_now        || exit 1
orc _gold_unc  --include-uncovered
orc _gold_shuf --shuffle-refs

if [ ! -f "$PAD_FROM" ]; then
  echo "### $(date '+%F %T')  pad source: rrf k=50 for $M"
  $P -u analysis/corpus_rag_run.py generate --pipeline rrf --top-k 50 \
     --corpora combined --models "$M" \
    || echo "### $(date '+%F %T')  FAILED pad-source generate"
fi
if [ -f "$PAD_FROM" ]; then
  orc _gold_pad --shuffle-refs --pad-junk 20 --pad-from "$PAD_FROM"
else
  echo "### $(date '+%F %T')  no pad source; _gold_pad skipped"
fi

orc _gold_cites --cites-only --cites-from "$D/oracle_${SLUG}_gold_now.csv"

echo "### $(date '+%F %T')  filling the other two free3 seats (FAU, free)"
$P -u analysis/oracle_experiment.py rejudge --judges free3 --arm gold \
  || echo '### rejudge exited non-zero'
echo "### $(date '+%F %T')  done -- put DeepSeek back in GEN in oracle_ablation_table.py"
