#!/usr/bin/env bash
# Regenerate the ls_text arms, non-DeepSeek generators first.
#
# The 09-11 `git clean -fd` deleted every rag_titled_* directory, so the
# Leitsatz arms behind the two-hop claim are gone along with the rest. The claim
# rested on DeepSeek alone even before that, which is why all seven generators
# are here.
#
# Generation only. There is no judge panel that excludes DeepSeek -- PANELS is
# free (Qwen3.6 + DeepSeek) and free3 (+ gpt-oss) -- and judge() walks its seats
# one at a time, so judging now would bank the two working seats and then spend
# the DeepSeek seat's full retry budget against a dead endpoint. That cost 1h41
# and 55min on the two LS arms it hit this morning. Six arms of that is ten
# hours for nothing. Generate now, judge in one pass when DeepSeek answers.
#
# STALE as of 8ba3fe0e: this said the arms cannot reproduce the published LS
# numbers because the query rewrite cache was deleted and qa.ReWriter samples.
# The caches were never actually lost -- CachedReWriter replays a tracked
# analysis/out/rewrites_*.json -- so these arms search with the published
# queries and the numbers are comparable after all.
#
#     setsid nohup bash analysis/run_ls_text_nods.sh > logs/ls_text_regen.log 2>&1 < /dev/null &
set -u
P=~/miniconda3/envs/plexam/bin/python
export PYTHONPATH=analysis

for M in Qwen/Qwen3.6-35B-A3B-FP8 \
         RedHatAI/gemma-4-31B-it-FP8-block \
         openai/gpt-oss-120b \
         qwen3-next-80b-a3b-instruct \
         GaleneAI/Magistral-Small-2509-FP8-Dynamic \
         RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-FP8; do
  echo "### $(date '+%F %T')  ls_text k=10 generate $M"
  $P -u analysis/corpus_rag_run.py generate --pipeline ls_text --top-k 10 \
     --corpora combined --models "$M" \
    || echo "### $(date '+%F %T')  FAILED $M"
done
echo "### $(date '+%F %T')  non-DeepSeek ls_text generation done"
