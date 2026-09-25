#!/usr/bin/env bash
# The DeepSeek oracle ablation, end to end, on the combined corpus.
#
# This is the arm set tab:oracle-ablation lost DeepSeek over: its conditions
# were the 08-20 run against the *federal* knowledge base -- 35.0 of the 51.9
# norms the solution cites per case, against 47.9 for the combined one -- so
# its row could not anchor on the same oracle arm the ladder prints, and the
# table shipped as two generators instead of three. The 09-11 `git clean -fd`
# then deleted every arm, federal ones included, so there is nothing left to be
# inconsistent with: everything below regenerates against
# my_knowledge_base_bayern_titled and the row becomes comparable by
# construction.
#
# Eight arms, in dependency order:
#
#   0  rrf k=50          NOT an oracle arm. _gold_pad draws its junk norms from
#                        a real pipeline arm, so the dilution condition is
#                        padded with plausible retrieved-but-not-cited norms
#                        rather than random ones. This is the one arm here that
#                        does NOT reproduce the published run: the query rewrite
#                        cache was deleted and qa.ReWriter samples, so the junk
#                        is a different draw. Junk is still junk and the
#                        condition tests dilution, not any particular norm.
#   1  _gold_combined    the anchor, and tab:ladder's DeepSeek oracle cell
#   2  _gold_now         not printed; _gold_cites lifts its norm list verbatim,
#                        so a wrong one here silently corrupts the leakage test
#   3  _gold_unc         + norms the corpus lacks, named but not supplied
#   4  _gold_shuf        solution-derived prominence order destroyed
#   5  _gold_pad         + 20 junk norms, interleaved; pairs against _gold_shuf
#   6  _gold_cites       the same norms named, all wording withheld
#   7  _gold_top10       selection cut to the ten most-cited
#
# Everything except arm 0 reproduces the published context exactly: the oracle
# does not retrieve, so the resampled rewrite cache never reaches it, and the
# coverage line still reads "47.9 of 51.9 cited norms per case".
#
# The gate probes with a realistic token count. A 120-token ping returned in
# 1.4s on 09-11 while every essay behind it hit the 600s timeout.
#
#     setsid nohup bash analysis/run_ds_ablation.sh > logs/ds_ablation.log 2>&1 < /dev/null &
set -u
P=~/miniconda3/envs/plexam/bin/python
export PYTHONPATH=analysis
M=deepseek-ai/DeepSeek-V4-Flash          # aliased to -0731 in endpoints.yaml
SLUG=${M//\//_}
D=zubaers_result/essay_writing/oracle_rag/ji2
J=Qwen/Qwen3.6-35B-A3B-FP8
PAD_FROM=zubaers_result/essay_writing/rag_titled_combined_rrf_k50/ji2/rag_${SLUG}.csv
PROBE_TOKENS=${PROBE_TOKENS:-1500}
PROBE_TIMEOUT=${PROBE_TIMEOUT:-180}

wait_up() {
  local n=0
  until $P analysis/endpoint_probe.py "$M" "$PROBE_TOKENS" "$PROBE_TIMEOUT" 2>&1 \
        | grep -q 'up,'; do
    n=$((n+1))
    [ $((n % 6)) -eq 1 ] && echo "### $(date '+%F %T')  $M not generating, waiting"
    sleep 300
  done
  echo "### $(date '+%F %T')  $M answering at generation length"
}

orc() {  # tag, extra args...
  local tag=$1; shift
  [ -f "$D/oracle_${SLUG}${tag}.csv" ] && {
    echo "### $(date '+%F %T')  $tag exists, skipping"; return 0; }
  local n
  for n in $(seq 1 24); do
    echo "### $(date '+%F %T')  $tag attempt $n/24"
    wait_up
    if timeout 3h $P -u analysis/oracle_experiment.py run --model "$M" \
         --tag "$tag" --judge "$J" --max-passages "${K:-120}" --chunk 5 "$@"; then
      return 0
    fi
    echo "### $(date '+%F %T')  $tag attempt $n exited $?"
    [ -f "$D/oracle_${SLUG}${tag}.csv.part" ] && echo "###   banked $(( $(wc -l < "$D/oracle_${SLUG}${tag}.csv.part") - 1 )) case(s)"
    sleep 120
  done
  echo "### $(date '+%F %T')  $tag FAILED after 24 attempts"; return 1
}

# --- arm 0: the pad source -------------------------------------------------
if [ ! -f "$PAD_FROM" ]; then
  echo "### $(date '+%F %T')  pad source: rrf k=50 for $M"
  wait_up
  $P -u analysis/corpus_rag_run.py generate --pipeline rrf --top-k 50 \
     --corpora combined --models "$M" \
    || echo "### $(date '+%F %T')  FAILED pad-source generate"
fi

# --- arms 1-7: the oracle family ------------------------------------------
orc _gold_combined || exit 1
orc _gold_now      || exit 1
orc _gold_unc   --include-uncovered
orc _gold_shuf  --shuffle-refs
if [ -f "$PAD_FROM" ]; then
  orc _gold_pad --shuffle-refs --pad-junk 20 --pad-from "$PAD_FROM"
else
  echo "### $(date '+%F %T')  no pad source; _gold_pad skipped"
fi
orc _gold_cites --cites-only --cites-from "$D/oracle_${SLUG}_gold_now.csv"
K=10 orc _gold_top10

echo "### $(date '+%F %T')  filling the other two free seats"
$P -u analysis/oracle_experiment.py rejudge --judges free3 --arm gold \
  || echo '### rejudge exited non-zero'
echo "### $(date '+%F %T')  done -- put DeepSeek back in GEN in oracle_ablation_table.py"
