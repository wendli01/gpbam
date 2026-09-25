#!/usr/bin/env bash
# Judge the Leitsatz arms. run_ls_arms.sh generated thirteen and judged none --
# it deliberately left judging to run_ds_seat_watch.sh, which never ran
# (logs/ls_seat_fill.log is still 0 bytes), so nine arms have sat as essays
# with no scores since 09-09.
#
#   ls_text k=10        6 arms  (DeepSeek was already judged)
#   cite_ls_rrf k=10    3 arms  (Magistral, Mistral, Qwen3-Next)
#   federal rrf         gpt-oss is missing only its DeepSeek seat
#
# Neither ls_text nor cite+LS+RRF at k=10 is a printed ladder column; they back
# the two-hop claim, which rested on DeepSeek alone. Sequential on purpose: all
# three free3 seats are FAU endpoints and judging runs three clients per arm as
# it is.
#
#     setsid nohup bash analysis/run_ls_judging.sh > logs/ls_judging.log 2>&1 < /dev/null &
set -u
P=~/miniconda3/envs/plexam/bin/python
export PYTHONPATH=analysis

j() {  # pipeline, top_k, model
  echo "### $(date '+%F %T')  judge $1 k=$2 $3"
  $P -u analysis/corpus_rag_run.py judge --pipeline "$1" --top-k "$2" \
     --corpora combined --judges free3 --yes --models "$3" \
    || echo "### $(date '+%F %T')  FAILED $1 k=$2 $3"
}

for M in RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-FP8 \
         GaleneAI/Magistral-Small-2509-FP8-Dynamic \
         qwen3-next-80b-a3b-instruct \
         Qwen/Qwen3.6-35B-A3B-FP8 \
         RedHatAI/gemma-4-31B-it-FP8-block \
         openai/gpt-oss-120b; do
  j ls_text 10 "$M"
done

for M in GaleneAI/Magistral-Small-2509-FP8-Dynamic \
         RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-FP8 \
         qwen3-next-80b-a3b-instruct; do
  j cite_ls_rrf 10 "$M"
done

echo "### $(date '+%F %T')  federal rrf gpt-oss, DeepSeek seat only"
$P -u analysis/corpus_rag_run.py judge --pipeline rrf --top-k 10 \
   --corpora federal --judges free3 --yes --models openai/gpt-oss-120b \
  || echo "### $(date '+%F %T')  FAILED federal rrf gpt-oss"

echo "### $(date '+%F %T')  all LS judging done"
