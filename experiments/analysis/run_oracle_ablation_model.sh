#!/usr/bin/env bash
# The oracle ablation for a second generator.
#
# Reviewer point: Table 6 decomposes the oracle advantage on DeepSeek alone,
# and DeepSeek is also a judge seat and the source of the headline. One more
# generator through the same five conditions removes that.
#
# The conditions, and what each one moves, exactly as the DeepSeek run:
#   _gold_now    the oracle itself -- the statutes the reference solution cites
#   _gold_unc    + norms the corpus lacks, named but not supplied
#   _gold_shuf   the solution-derived prominence order destroyed
#   _gold_pad    + 20 non-cited norms, interleaved; pairs against _gold_shuf,
#                never against _gold_now, because padding shuffles as it
#                interleaves and order has to be held constant across the pair
#   _gold_cites  the same norms named, all wording withheld -- the leakage test
#
# --max-passages 120 is above every per-case gold count (max 97), so the cap
# never binds and the arms differ only in what they are meant to differ in.
#
# No GPU: the oracle takes its norms from the solution and no retriever is
# loaded, so this runs beside anything holding the card.
#
#     MODEL=RedHatAI/gemma-4-31B-it-FP8-block bash analysis/run_oracle_ablation_model.sh
#
# On a degraded endpoint, supervise each arm and let it resume:
#
#     MODEL=... CHUNK=5 LIMIT=45m ATTEMPTS=20 bash analysis/run_oracle_ablation_model.sh
set -u
P=~/miniconda3/envs/plexam/bin/python
export PYTHONPATH=analysis
M=${MODEL:?set MODEL to the endpoint id of the generator}
J=Qwen/Qwen3.6-35B-A3B-FP8
D=zubaers_result/essay_writing/oracle_rag/ji2
K=120
#: cases per .part checkpoint, and the per-attempt wall clock. Defaults suit a
#: healthy endpoint; for a degraded one lower CHUNK so an attempt banks.
CHUNK=${CHUNK:-5}
LIMIT=${LIMIT:-24h}
ATTEMPTS=${ATTEMPTS:-1}
SLUG=${M//\//_}
PAD_FROM=zubaers_result/essay_writing/rag_titled_combined_rrf_k50/ji2/rag_${SLUG}.csv

for W in ${WAIT:-}; do
  echo "### $(date '+%F %T')  waiting on pid $W"
  while kill -0 "$W" 2>/dev/null; do sleep 60; done
done

# ATTEMPTS>1 supervises each arm under `timeout` and retries. Generation
# checkpoints to a .part every CHUNK cases and resumes from it, so an attempt
# that is cut off still advances the arm -- which is the whole point, and was
# not true when CHUNK was 20 and the cap 45 minutes: twelve retries of the
# top-10 arm on 09-10 wrote no .part at all and each one restarted from zero.
# Keep CHUNK*per-case well under LIMIT or the same thing happens again.
run() {  # tag, extra args...
  local tag=$1; shift
  if [ -f "$D/oracle_${SLUG}${tag}.csv" ]; then
    echo "### $(date '+%F %T')  $tag exists, skipping"; return 0
  fi
  local n
  for n in $(seq 1 "$ATTEMPTS"); do
    echo "### $(date '+%F %T')  $tag attempt $n/$ATTEMPTS"
    if timeout "$LIMIT" $P -u analysis/oracle_experiment.py run \
         --model "$M" --tag "$tag" --judge "$J" --max-passages "$K" \
         --chunk "$CHUNK" "$@"; then
      return 0
    fi
    echo "### $(date '+%F %T')  $tag attempt $n exited $?"
    [ -f "$D/oracle_${SLUG}${tag}.csv.part" ] \
      && echo "###   banked $(( $(wc -l < "$D/oracle_${SLUG}${tag}.csv.part") - 1 )) case(s)"
    sleep "${BACKOFF:-120}"
  done
  echo "### $(date '+%F %T')  $tag failed after $ATTEMPTS attempt(s)"
  return 1
}

run _gold_now                                              || exit 1
run _gold_unc   --include-uncovered
run _gold_shuf  --shuffle-refs
if [ -f "$PAD_FROM" ]; then
  run _gold_pad --shuffle-refs --pad-junk 20 --pad-from "$PAD_FROM"
else
  echo "### no rrf_k50 arm at $PAD_FROM; skipping _gold_pad (its junk source)"
fi
run _gold_cites --cites-only --cites-from "$D/oracle_${SLUG}_gold_now.csv"

echo "### $(date '+%F %T')  third free seat on the new arms"
$P -u analysis/oracle_experiment.py rejudge --judges free3 --arm gold \
  || echo '### rejudge exited non-zero; the seat watcher will fill it'
echo "### $(date '+%F %T')  ablation for $M done"
