#!/usr/bin/env bash
# Judge the six ls_text arms in two passes, around the DeepSeek deployment.
#
# Replaces run_ls_text_judge_watch.sh, which deadlocked. Its window_open() held
# judging whenever `oracle_experiment.py run` was alive, on the reasoning that
# the oracle job owns the DeepSeek seat. True -- but the FAU ablation runs for
# ten hours per arm and there are seven arms, so the watcher would have waited
# past the deadline with five generated arms unjudged.
#
# Two passes instead:
#   1. --judges nods   Qwen3.6 + gpt-oss, neither of which the ablation touches.
#                      Banks two of the three free3 seats immediately.
#   2. --judges free3  once no oracle run is alive, tops up DeepSeek only.
#                      judge() fills only the seats a file lacks, so pass 1 is
#                      not redone.
#
# pt.med() needs all three seats and returns None otherwise, so an arm is not
# reportable until pass 2 reaches it. Pass 1 still removes ~2/3 of the wait.
#
#     setsid nohup bash analysis/run_ls_text_judge.sh > logs/ls_text_judge2.log 2>&1 < /dev/null &
set -u
P=~/miniconda3/envs/plexam/bin/python
export PYTHONPATH=analysis
MODELS="Qwen/Qwen3.6-35B-A3B-FP8 RedHatAI/gemma-4-31B-it-FP8-block \
        openai/gpt-oss-120b qwen3-next-80b-a3b-instruct \
        GaleneAI/Magistral-Small-2509-FP8-Dynamic \
        RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-FP8"

pass() {   # panel
  for m in $MODELS; do
    echo "### $(date '+%F %T')  [$1] $m"
    $P -u analysis/corpus_rag_run.py judge --pipeline ls_text --top-k 10 \
       --corpora combined --judges "$1" --yes --models "$m" \
      || echo "### $(date '+%F %T')  FAILED [$1] $m"
  done
}

echo "### $(date '+%F %T')  pass 1: Qwen3.6 + gpt-oss (DeepSeek left alone)"
pass nods

echo "### $(date '+%F %T')  waiting for the DeepSeek deployment"
while pgrep -f 'oracle_experiment[.]py run' > /dev/null; do sleep 300; done
echo "### $(date '+%F %T')  pass 2: topping up the DeepSeek seat"
pass free3
echo "### $(date '+%F %T')  done"
