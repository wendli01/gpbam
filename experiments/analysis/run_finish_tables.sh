#!/usr/bin/env bash
# The two cells that stand between the tables and done.
#
#   tab:ladder          Mistral-Small-3.2 cite+LS+RRF_50 -- essays generated
#                       09-09 10:37, never judged. run_ds_seat_watch.sh was
#                       supposed to pick it up and never ran (ls_seat_fill.log
#                       is 0 bytes), so the cell is blank for want of judging,
#                       not for want of an arm.
#   tab:oracle-ablation DeepSeek top-10 -- the file does not exist. The backend
#                       refused connections when run_oracle_top10.sh reached it
#                       on 09-08 and the call is still commented out there.
#                       DeepSeek is answering again (federal_rrf is drawing
#                       ~248 s/batch off it right now), so it can be run.
#
# Both wait on the federal-RRF chain. Every step below is a DeepSeek client --
# the judge ensemble's second seat is DeepSeek-V4-Flash, and the top-10 arm
# generates on it -- and federal_rrf is generating on the same endpoint. Three
# clients on that endpoint took 43 minutes to produce zero essays on 09-07 and
# dragged the judges from ~30 s to ~300 s per essay at the same time. Serial is
# faster than parallel here; it has been measured twice.
#
# Free: FAU generation, free3 judging. No nano, nothing billed.
#
#     setsid nohup bash analysis/run_finish_tables.sh > logs/finish_tables.log 2>&1 < /dev/null &
set -u
P=~/miniconda3/envs/plexam/bin/python
export PYTHONPATH=analysis
WAIT=${WAIT:-1744754}

for pid in $WAIT; do
  while kill -0 "$pid" 2>/dev/null; do sleep 120; done
  echo "### $(date '+%F %T')  pid $pid drained"
done

echo "### $(date '+%F %T')  judging Mistral cite+LS+RRF_50 (completes tab:ladder)"
$P -u analysis/corpus_rag_run.py judge --pipeline cite_ls_rrf --top-k 50 \
   --corpora combined --judges free3 --yes \
   --models RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-FP8 \
  || echo "### $(date '+%F %T')  FAILED Mistral cite_ls_rrf50 judging"

# oracle_experiment's --judge takes one model id, not a scale name: `run`
# scores with a single seat and `rejudge --judges free3` fills the rest. Same
# two steps run_oracle_top10.sh uses, and rejudge is incremental, so the second
# call touches only the file the first one wrote.
echo "### $(date '+%F %T')  DeepSeek _gold_top10 (completes tab:oracle-ablation)"
$P -u analysis/oracle_experiment.py run --model deepseek-ai/DeepSeek-V4-Flash \
   --tag _gold_top10 --judge Qwen/Qwen3.6-35B-A3B-FP8 --max-passages 10 \
  || echo "### $(date '+%F %T')  FAILED DeepSeek _gold_top10"

echo "### $(date '+%F %T')  filling the remaining free3 seats"
$P -u analysis/oracle_experiment.py rejudge --judges free3 --arm other \
  || echo "### $(date '+%F %T')  FAILED rejudge"

echo "### $(date '+%F %T')  done; rebuild both tables"
