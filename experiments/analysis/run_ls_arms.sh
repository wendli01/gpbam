#!/usr/bin/env bash
# The Leitsatz arms the ladder is missing, for every model that lacks them.
#
# Why: the two-hop story rested on one generator. ls_text existed for
# DeepSeek-V4-Flash alone (1 of 8), cite+LS+RRF at k=10 for 4 of 8, and the
# printed cite+LS+RRF_50 column for 6 of 8 -- so "resolving the Normenkette of
# a retrieved headnote is the best non-oracle arm" was a claim about DeepSeek
# with a little corroboration, while every other column spoke for eight models.
#
# Thirteen arms:
#   cite_ls_rrf k=50   Mistral-Small-3.2, Gemma-4-E4B   (completes tab:ladder)
#   cite_ls_rrf k=10   Qwen3-Next, Mistral-Small, Magistral-Small, Gemma-4-E4B
#   ls_text     k=10   the seven models that are not DeepSeek
#
# Generation only. Judging is left to run_ds_seat_watch.sh afterwards, because
# oracle_experiment.py rejudge is holding the DeepSeek and gpt-oss judge seats
# for ~15h and a second client on either one makes both slower than running
# them in sequence -- measured, twice, on 09-07.
#
# TWO chains, not one per model. Retrieval runs locally and each generate
# process loads its own embedder onto a single 7.9 GiB card; five concurrent
# chains took ~1.5 GB each and every one died with CUBLAS_STATUS_ALLOC_FAILED
# before writing a line. Two fit with headroom, and more would not help:
# Gemma-4-E4B alone sets the wall clock at 247 s/essay -- 5.5h per arm, three
# arms -- against ~25 s/essay for Mistral-Small and Magistral-Small, so every
# other model finishes inside its shadow either way.
#
#     bash analysis/run_ls_arms.sh
set -u
P=~/miniconda3/envs/plexam/bin/python
export PYTHONPATH=analysis

gen() {  # pipeline, top_k, model
  echo "### $(date '+%F %T')  gen $1 k=$2 $3"
  $P -u analysis/corpus_rag_run.py generate --pipeline "$1" --top-k "$2" \
     --corpora combined --models "$3" \
    || echo "### $(date '+%F %T')  FAILED $1 k=$2 $3"
}

E4B=google/gemma-4-E4B-it
MIS=RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-FP8
MAG=GaleneAI/Magistral-Small-2509-FP8-Dynamic
QN=qwen3-next-80b-a3b-instruct

chain_slow() {
  gen cite_ls_rrf 50 "$E4B"
  gen cite_ls_rrf 10 "$E4B"
  gen ls_text     10 "$E4B"
}

chain_fast() {
  gen cite_ls_rrf 50 "$MIS"
  gen cite_ls_rrf 10 "$MIS"
  gen ls_text     10 "$MIS"
  gen cite_ls_rrf 10 "$MAG"
  gen ls_text     10 "$MAG"
  gen cite_ls_rrf 10 "$QN"
  gen ls_text     10 "$QN"
  gen ls_text     10 Qwen/Qwen3.6-35B-A3B-FP8
  gen ls_text     10 RedHatAI/gemma-4-31B-it-FP8-block
  gen ls_text     10 openai/gpt-oss-120b
}

chain_slow & S=$!
chain_fast & F=$!
echo "### $(date '+%F %T')  two chains: slow=$S (Gemma-4-E4B, ~16h) fast=$F (10 arms, ~5h)"
wait $S $F
echo "### $(date '+%F %T')  all LS generation done; seats left to run_ds_seat_watch.sh"
