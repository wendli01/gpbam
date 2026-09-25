#!/usr/bin/env bash
# Mirror the un-committable artefacts to a directory OUTSIDE the repo.
#
# The results tree and the knowledge bases are gitignored on purpose (they are
# too big for the history, and CLAUDE.md requires they stay ignored). Ignoring
# them stops `git clean -fd`, which is what destroyed every arm on 09-11 -- but
# it does not stop `git clean -fdx`, an `rm -rf` in the repo, or a stray script.
# Anything living only inside the working tree is one command from gone.
#
# This is a same-disk copy, so it is protection against mistakes, not against
# drive failure. For that the mirror needs to leave the machine.
#
# One-shot:   bash analysis/backup_results.sh
# Every 30m:  setsid nohup bash analysis/backup_results.sh --loop \
#                 > logs/backup.log 2>&1 < /dev/null &
set -u
SRC=${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
DST=${DST:-$HOME/gpbam-backup}
mkdir -p "$DST"

sync_once() {
  echo "### $(date '+%F %T')  syncing -> $DST"
  # --delete is deliberately ABSENT: a deletion in the working tree must not
  # propagate to the mirror, because deletion in the working tree is the exact
  # failure this guards against.
  rsync -a --info=stats2 \
    "$SRC/zubaers_result" "$SRC/analysis/out" "$SRC/logs" "$DST/" \
    | grep -E 'Number of regular files transferred|Total transferred file size'
  echo "### $(date '+%F %T')  $(du -sh "$DST" | cut -f1) mirrored"
}

if [ "${1:-}" = --loop ]; then
  while true; do sync_once; sleep 1800; done
else
  sync_once
fi
