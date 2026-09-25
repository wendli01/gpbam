#!/usr/bin/env bash
# Everything that has to happen after the corpus rebuild, in order, unattended.
#
#   1. wait for the running rebuild_kb_with_titles.py to exit
#   2. preflight: are both rebuilt corpora actually complete and queryable?
#   3. retrieval quality on both, recall@10 and recall@50   (no API calls)
#   4. generate + judge the retrieval arm for the five free models
#
# Every stage is idempotent -- each underlying script reuses whatever it already
# wrote -- so the safe response to any interruption is to run this again.  It is
# deliberately not `set -e`: a dead endpoint in stage 4 should cost one model,
# not the queue.  Stage 2 is the exception, because running retrieval against a
# half-written corpus produces numbers that look plausible and mean nothing.
#
# Launch detached, so closing the terminal cannot kill it:
#
#     cd experiments
#     setsid nohup bash analysis/run_queue.sh > logs/queue.log 2>&1 < /dev/null &
#
set -uo pipefail

PY=${PY:-$HOME/miniconda3/envs/plexam/bin/python}
WAIT_PID=${WAIT_PID:-}                     # rebuild process to wait for; empty = don't wait
SRC_FED=${SRC_FED:-./my_knowledge_base}
SRC_COMBINED=${SRC_COMBINED:-./my_knowledge_base_bayern}
FED=${FED:-./my_knowledge_base_titled}
COMBINED=${COMBINED:-./my_knowledge_base_bayern_titled}
ONLY=${ONLY:-}                             # 'preflight' stops after the checks
KB=${KB:-$COMBINED}                        # corpus the generation arm retrieves from
TOP_K=${TOP_K:-10}
GATE=${GATE:-0}                            # stop before stage 4 if recall@10 < GATE (%)
LOGS=${LOGS:-logs}

mkdir -p "$LOGS"
say() { echo -e "\n[$(date '+%F %T')] $*" ; }

# --- 1. wait ---------------------------------------------------------------
if [[ -n "$WAIT_PID" ]]; then
  say "waiting for the corpus rebuild (pid $WAIT_PID)"
  while [[ -d /proc/$WAIT_PID ]]; do sleep 60; done
  say "rebuild process exited"
fi

# --- 2. preflight ----------------------------------------------------------
# A rebuild that died halfway leaves a table that opens fine and is simply
# short, and a table whose FTS index was never built silently loses the lexical
# half of hybrid search.  Both are checked here rather than discovered in the
# numbers three hours later.
say "preflight"
"$PY" - "$FED" "$COMBINED" "$SRC_FED" "$SRC_COMBINED" <<'EOF'
import sys, lancedb
dsts, srcs = sys.argv[1:3], sys.argv[3:5]
# the rebuild copies rows one-for-one, so the source table is the row count to
# expect -- no constant to go stale when the corpus is refetched
expect = {s: lancedb.connect(s).open_table('documents').count_rows() for s in srcs}
bad = False
for dst, src in zip(dsts, srcs):
    try:
        t = lancedb.connect(dst).open_table('documents')
    except Exception as e:
        print(f'  FAIL {dst}: {e}'); bad = True; continue
    n, cols = t.count_rows(), t.schema.names
    # An FTS index does not show up in list_indices() in this LanceDB version, so
    # ask the question the pipeline actually asks: one hybrid search, exactly as
    # LawRetriever issues it.  That exercises the vectors, the full-text index and
    # the registered embedding function together, and a missing piece raises here
    # instead of quietly halving the search three hours later.
    try:
        hits = len(t.search('beschlagnahme', query_type='hybrid').limit(3).to_list())
    except Exception as e:
        hits, err = 0, e
    else:
        err = None
    ok = n == expect[src] and 'embed_text' in cols and hits > 0
    print(f'  {"OK  " if ok else "FAIL"} {dst}: {n}/{expect[src]} rows, '
          f'embed_text={"embed_text" in cols}, hybrid search -> {hits} rows'
          + (f' ({type(err).__name__}: {err})' if err else ''))
    bad |= not ok
sys.exit(1 if bad else 0)
EOF
if [[ $? -ne 0 ]]; then
  say "PREFLIGHT FAILED -- the corpora are not complete. Nothing else will run."
  exit 1
fi
[[ "$ONLY" == preflight ]] && { say "preflight only -- stopping here"; exit 0; }

# --- 3. retrieval quality --------------------------------------------------
# Zero API calls: the query rewrites are cached from the earlier run, which is
# also what keeps this comparable to the pre-rebuild numbers.
for k in 10 50; do
  say "retrieval quality, recall@$k"
  PYTHONPATH=analysis "$PY" -u analysis/corpus_comparison.py \
      --top-k "$k" --queries rewriter --titled \
      --kb-federal "$FED" --kb-bayern "$COMBINED" --tag _titled \
      2>&1 | tee "$LOGS/recall_k${k}_titled.log"
done

# --- 4. gate ---------------------------------------------------------------
R10=$(PYTHONPATH=analysis "$PY" -c "
import pandas as pd
d = pd.read_csv('analysis/out/corpus_comparison_k10_full_rewriter_titled.csv')
print(round(100 * d.recall.max(), 2))
" 2>/dev/null || echo 0)
# For reference, before the rebuild the same cached rewrites scored recall@50 of
# 2.28% (federal) and 2.93% (combined) -- with jina queries against bge vectors,
# so that is the noise floor rather than a real retrieval result.
say "best recall@10 after the rebuild: ${R10}%  (pre-rebuild recall@50 was 2.3% / 2.9%)"
if (( $(echo "$R10 < $GATE" | bc -l) )); then
  say "below the gate of ${GATE}% -- stopping before generation. Re-run with GATE=0 to force."
  exit 2
fi

# --- 5. generate + judge ---------------------------------------------------
# Both corpora, so the score side answers the same question the recall side
# does: 5 models x 81 cases x 2 corpora, all on free endpoints.
say "generation + judging over both corpora (top_k=$TOP_K), five free models, two free judges"
PYTHONPATH=analysis "$PY" -u analysis/corpus_rag_run.py run \
    --top-k "$TOP_K" 2>&1 | tee "$LOGS/corpus_rag_run.log"

say "queue finished"
