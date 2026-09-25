#!/usr/bin/env bash
# What we are asking of NHR@FAU right now, once every five minutes.
#
# The deployment delivers ~600 tok/s no matter how many streams we open, so
# extra concurrency only queues -- ours and everyone else's. This does not kill
# anything (killing is what creates orphans: FAU finishes an abandoned
# generation anyway, so a kill wastes the work and keeps the load). It records
# the footprint so a duplicate-spawning bug shows up as a connection count
# above the streams the running commands actually declared.
#
#     setsid nohup bash analysis/fau_load_watch.sh > /dev/null 2>&1 < /dev/null &
set -u
OUT=logs/fau_load.log
BUDGET=${BUDGET:-55}

while :; do
  tot=0; detail=''
  for p in $(pgrep -f 'oracle_experiment.py run|corpus_rag_run.py'); do
    c=$(ss -tnp 2>/dev/null | grep -c "pid=$p,")
    [ "$c" -gt 0 ] || continue
    tag=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null \
          | sed -E 's/.*--pipeline rrf.*/pad-source/;s/.*--tag ([^ ]+).*/\1/;s/.*judge .*--models ([^ ]+).*/judge:\1/' \
          | cut -c1-26)
    detail="$detail ${tag}=$c"; tot=$((tot + c))
  done
  warn=''
  [ "$tot" -gt "$BUDGET" ] && warn="  !! over budget ($BUDGET)"
  echo "$(date '+%F %T')  streams=$tot$detail$warn" >> $OUT
  sleep 300
done
