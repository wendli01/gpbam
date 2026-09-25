#!/usr/bin/env bash
# The DeepSeek oracle ablation on NHR@FAU, every independent arm at once.
#
# Replaces run_ds_ablation_fau.sh, which ran the arms one after another at
# ~10 h each. That pace was mostly self-inflicted -- the 300 s client timeout
# turned each essay into a new generation every five minutes (README, "Never
# let a client time out on NHR@FAU") -- and with that fixed the arms run side by
# side. Priority goes to the two cells the paper is waiting on: _gold_combined
# (the anchor every other column is read against) and _gold_top10 (the cell
# tab:oracle-ablation prints as "--") get 10 streams each, the rest 5.
#
# _gold_now is gone. With oracle_experiment.KB now the combined corpus it is
# the same configuration as _gold_combined, byte for byte, and the FAU gateway
# caches responses by request -- it would have been a copy, not a replicate.
# _gold_cites therefore takes its citations from _gold_combined, and waits for it.
#
# An arm whose attempt is already running (from the old supervisor) is waited
# on, not started twice: two writers on one .part file would corrupt it.
#
#     setsid nohup bash analysis/run_ds_ablation_parallel.sh > logs/ds_ablation_par.log 2>&1 < /dev/null &
set -u
P=~/miniconda3/envs/plexam/bin/python
export PYTHONPATH=analysis
M=deepseek-ai/DeepSeek-V4-Flash          # endpoints.yaml aliases this to -0731
SLUG=${M//\//_}
D=zubaers_result/essay_writing/oracle_rag/ji2
J=Qwen/Qwen3.6-35B-A3B-FP8
PAD_FROM=zubaers_result/essay_writing/rag_titled_combined_rrf_k50/ji2/rag_${SLUG}.csv

log() { echo "### $(date '+%F %T')  $*"; }

# A short probe with a generous timeout: under our own 40-odd streams a
# single request is slow, and that is load, not an outage.
wait_up() {
  until $P analysis/endpoint_probe.py "$M" 120 300 2>&1 | grep -q 'up,'; do
    sleep 300
  done
}

orc() {  # tag, streams, extra args...
  local tag=$1 conc=$2; shift 2
  local f="$D/oracle_${SLUG}${tag}.csv" n
  for n in $(seq 1 12); do
    while pgrep -f -- "tag ${tag} " > /dev/null; do sleep 60; done
    [ -f "$f" ] && { log "$tag done"; return 0; }
    log "$tag attempt $n/12 ($conc streams)"
    wait_up
    timeout 6h $P -u analysis/oracle_experiment.py run --model "$M" --tag "$tag" \
      --judge "$J" --max-passages "${K:-120}" --chunk "$conc" --concurrency "$conc" "$@" \
      && { log "$tag done"; return 0; }
    log "$tag attempt $n failed"
    sleep 120
  done
  log "$tag FAILED after 12 attempts"; return 1
}

pad() {
  local n
  for n in 1 2 3; do
    [ -f "$PAD_FROM" ] && break
    log "pad source (RRF k=50) attempt $n/3"
    wait_up
    $P -u analysis/corpus_rag_run.py generate --pipeline rrf --top-k 50 \
       --corpora combined --models "$M" || log "pad source attempt $n failed"
  done
  [ -f "$PAD_FROM" ] || { log "pad source FAILED"; return 1; }
  orc _gold_pad 5 --shuffle-refs --pad-junk 20 --pad-from "$PAD_FROM"
}

# ARMS picks a subset, e.g. ARMS="combined top10" to start the priority pair
# first; a second instance with the rest is safe, since orc() waits on any
# attempt already running for its tag and skips a finished arm.
for a in ${ARMS:-combined top10 unc shuf pad}; do
  case $a in
    combined) ( orc _gold_combined 10 &&
                orc _gold_cites 5 --cites-only \
                    --cites-from "$D/oracle_${SLUG}_gold_combined.csv" ) & ;;
    top10)    K=10 orc _gold_top10 10 & ;;
    unc)      orc _gold_unc  5 --include-uncovered & ;;
    shuf)     orc _gold_shuf 5 --shuffle-refs & ;;
    pad)      pad & ;;
  esac
done
wait
[ -n "${ARMS:-}" ] && { log "subset done: $ARMS"; exit 0; }

log "filling the other two free3 seats"
# --arm other: arm_of() files _gold_combined.csv under `other`, not `gold`
for arm in gold other; do
  $P -u analysis/oracle_experiment.py rejudge --judges free3 --arm "$arm" --yes \
    || log "rejudge --arm $arm exited non-zero"
done
log "done"
