#!/usr/bin/env bash
# The federal-corpus essay arms at RRF, so tab:corpus reads at one fusion.
#
# The corpus table contrasts the federal knowledge base against the combined
# one. Its retrieval rows already sit at RRF -- recall_battery grew a
# corpus_federal_rrf config for exactly that, and the combined column there is
# literally tab:ladder's own RRF cell. Its essay rows do not: rag_titled_federal
# is the only federal generation that exists and it was produced under the
# round-robin sub-query merge, a fusion PAPER_COLS never prints. So the table
# currently measures its two halves at two settings and says so in the caption.
# These five arms remove the clause.
#
# Five models, the ones with a federal row in the table. Mistral-Small and
# Magistral-Small have no federal arm at all and are dashes on that side.
#
# Expect the story not to move, and that is the point: federal recall rose from
# 1.38 to 2.28 when the fusion improved, but federal state-law recall is 0.00
# under either, because that corpus contains no Bavarian law. The zero is the
# row the table exists to show and no merge rule can shift it. What changes is
# that both halves are then read at the same setting.
#
# Queued, not concurrent: run_ls_arms.sh holds the GPU embedder and
# oracle_experiment.py rejudge holds the DeepSeek and gpt-oss judge seats.
# Three clients on one endpoint is slower than the same work in sequence --
# measured twice on 09-07.
#
#     bash analysis/run_federal_rrf.sh
set -u
P=~/miniconda3/envs/plexam/bin/python
export PYTHONPATH=analysis

for W in ${WAIT:-}; do
  echo "### $(date '+%F %T')  waiting on pid $W"
  while kill -0 "$W" 2>/dev/null; do sleep 120; done
done

for M in deepseek-ai/DeepSeek-V4-Flash Qwen/Qwen3.6-35B-A3B-FP8 \
         openai/gpt-oss-120b RedHatAI/gemma-4-31B-it-FP8-block \
         qwen3-next-80b-a3b-instruct; do
  echo "### $(date '+%F %T')  generate rrf federal $M"
  $P -u analysis/corpus_rag_run.py generate --pipeline rrf --top-k 10 \
     --corpora federal --models "$M" \
    || { echo "### $(date '+%F %T')  FAILED generate $M"; continue; }
  echo "### $(date '+%F %T')  judge rrf federal $M"
  $P -u analysis/corpus_rag_run.py judge --pipeline rrf --top-k 10 \
     --corpora federal --models "$M" --judges free3 \
    || echo "### $(date '+%F %T')  FAILED judge $M"
done
echo "### $(date '+%F %T')  federal RRF arms done -> rebuild ladder_table.py"
