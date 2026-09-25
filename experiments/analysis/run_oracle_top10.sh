#!/usr/bin/env bash
# The short oracle: the ten most-cited gold norms instead of all of them.
#
# Reviewer point: the oracle's +3.40 is a net of "the right norms help" minus
# "a 41k-character context hurts", and tab:oracle-ablation prices the second
# half at -2.28 for +19.7 junk norms. If that slope applies to the oracle's own
# bulk, the informational value of the statutes is materially larger than the
# headline and the paper's claim -- access is not the binding constraint --
# does not follow. The arm that settles it is the same oracle with the context
# cut to the ten norms the reference solution leans on hardest.
#
# gold_sets() already ranks each case's norms by citation count, and
# OracleRetriever truncates that ranking at --max-passages, so the manipulation
# is one flag. The cap binds on all 81 cases: 10 of 47.4 supplied norms, 22% of
# the gold set, about a fifth of the context length.
#
# Paired against oracle_<model>_gold_combined.csv, NOT the _gold_now ablation
# arms: those were generated against ./my_knowledge_base when it was the
# federal corpus, and neither that corpus nor the landesrecht.json of the time
# can be reproduced (35.0 norms per case then, 47.6 now), so pairing there
# would move the corpus alongside the cap. The _gold_combined arms are today's
# corpus, carry all three free seats, and are the oracle column of
# Table~\ref{tab:ladder} -- i.e. the number the review is actually about.
#
# Two generators, because one is a quirk: DeepSeek is the headline, gpt-oss is
# a generator that is not also the source of the claim.
#
# No GPU and no billed judge: the oracle takes its norms from the solution, and
# free3 is Qwen + DeepSeek + gpt-oss.
#
#     bash analysis/run_oracle_top10.sh
set -u
P=~/miniconda3/envs/plexam/bin/python
export PYTHONPATH=analysis
J=Qwen/Qwen3.6-35B-A3B-FP8
D=zubaers_result/essay_writing/oracle_rag/ji2
K=10

# gpt-oss first, DeepSeek last, and DeepSeek only once the judge queues drain.
# A first attempt ran DeepSeek generation while two DeepSeek judge jobs held the
# same endpoint: 43 minutes for zero essays, and the judges fell from ~30s to
# ~300s per essay at the same time. Three clients on one model is slower in
# total than the same work serialised, so the arm that shares a model with a
# running job waits for it. WAIT holds the pids to drain first.
run_arm() {
  local M=$1
  local SLUG=${M//\//_}
  if [ -f "$D/oracle_${SLUG}_gold_top10.csv" ]; then
    echo "### $(date '+%F %T')  $SLUG _gold_top10 exists, skipping"; return 0
  fi
  echo "### $(date '+%F %T')  $SLUG _gold_top10"
  $P -u analysis/oracle_experiment.py run --model "$M" --tag _gold_top10 \
     --judge "$J" --max-passages "$K" \
    || echo "### $(date '+%F %T')  $SLUG failed"
}

run_arm openai/gpt-oss-120b

# Gemma-4-31B stands in for DeepSeek while the DeepSeek backend is degraded
# (10.28.52.159:8101 refused connections at 02:08 on 09-08; 4 essays in 2h57m
# before that, 10 min/essay after it came back -- 13h+ for one arm). It is also
# the better pairing: its five ablation conditions were all generated on
# 09-08 against today's corpus, so its own junk-norm slope and this arm are
# measured on the same footing, and the reviewer's arithmetic can be done
# end to end within one model. DeepSeek still wants this arm -- -2.28 is its
# number -- and goes back in the queue when its endpoint recovers.
run_arm RedHatAI/gemma-4-31B-it-FP8-block

for W in ${WAIT:-}; do
  echo "### $(date '+%F %T')  waiting on pid $W before the DeepSeek arm"
  while kill -0 "$W" 2>/dev/null; do sleep 60; done
done
# DeepSeek is commented out, not deleted. Its backend has been serving this
# arm at 600-3100 s per essay since 09-07 (5 essays in 3h12m on the last
# attempt, 4 in 2h57m before that, one outright connection refusal), against
# ~40 s for gpt-oss and Gemma on the same task. Two attempts have been killed.
# Uncomment when the endpoint is healthy again -- -2.28 is DeepSeek's number
# and the reviewer's arithmetic is about DeepSeek, so this arm is still wanted.
# run_arm deepseek-ai/DeepSeek-V4-Flash

# The two remaining free seats. --arm other and not --arm gold: arm_of() calls
# anything past `_gold.csv` 'other', so --arm gold never reaches a tagged
# ablation arm -- which is also why the gpt-oss ablation arms are still sitting
# on one seat. This pass fills those too.
echo "### $(date '+%F %T')  free3 seats on the tagged arms"
$P -u analysis/oracle_experiment.py rejudge --judges free3 --arm other \
  || echo '### rejudge exited non-zero'
echo "### $(date '+%F %T')  done"
