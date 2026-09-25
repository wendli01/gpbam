#!/usr/bin/env bash
# The oracle as it should always have been: over the corpus the retrieval arms
# actually search.
#
# oracle_experiment.KB used to be ./my_knowledge_base, which is federal-only,
# patched with a 100-norm landesrecht.json holding no GO and no BV -- two of the
# four most-cited Bavarian books, at 293 and 253 gold citations. The upper bound
# was therefore handicapped on precisely the axis the corpus section argues is
# binding: 35.0 of 51.9 cited norms per case and 34.4% of the state-law ones.
# Over my_knowledge_base_bayern_titled it is 47.4 and 94.9%, which also beats
# the stored _gold arms' 45.4 and 90.9%.
#
# A new tag rather than a regenerated _gold_now, for two reasons. The stored
# _gold arms cannot be reproduced at all, so overwriting nothing there is
# possible; and prompt_pilot pairs every oracle ablation -- shuffled references,
# padded junk, citations-only, qa2 -- against _gold_now, so replacing that file
# would silently re-base four published contrasts against a different oracle.
#
# Free: one FAU judge seat during generation, the second added by the rejudge
# pass. gpt-5-nano is not bought here; the ladder table these fill is on the
# free scale.
#
# --arm other on the rejudge, not --arm gold: arm_of() keys on the filename
# suffix, so _gold_combined.csv is filed under `other` and --arm gold silently
# matches nothing.
#
# Serial. API-only -- the norms come from the reference solution, so no
# retriever loads and the GPU stays with the ladder arms.
set -u
P=~/miniconda3/envs/plexam/bin/python
export PYTHONPATH=analysis
J=Qwen/Qwen3.6-35B-A3B-FP8
K=120

for m in deepseek-ai/DeepSeek-V4-Flash Qwen/Qwen3.6-35B-A3B-FP8 \
         openai/gpt-oss-120b qwen3-next-80b-a3b-instruct gemma4-31b-it; do
  echo "### $(date '+%F %T')  oracle _gold_combined / $m"
  $P -u analysis/oracle_experiment.py run --model "$m" --tag _gold_combined \
     --judge "$J" --max-passages "$K" \
    || echo "### $m failed; continuing"
done

echo "### $(date '+%F %T')  second free seat"
$P -u analysis/oracle_experiment.py rejudge --judges free --arm other \
  || echo '### rejudge exited non-zero'
echo "### $(date '+%F %T')  combined-corpus oracle arms done"
