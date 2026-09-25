"""Generate the essays an outage left blank in an oracle arm, and graft them in.

Neither of the two mechanisms that look like they cover this actually does.
``run_ds_ablation_fau.sh``'s ``orc()`` skips a tag whose final CSV exists, and
``generate_chunked`` resumes from a ``.part`` file it deletes on completion.
Both assume a failed arm *crashed*. On 2026-09-12 the FAU DeepSeek deployment
instead started returning empty answers, so ``_gold_cites`` and ``_gold_pad``
were written out as complete 81-row files with 21 and 26 blank tails -- nothing
downstream will ever refill them, and re-running ``run`` would regenerate all 81
and discard the essays that did land.

This rebuilds the arm's retriever from the same flags, generates only the rows
whose ``answer`` is blank, and writes those rows back into the existing file. The
arms are deterministic given their sources, so the top-up rows are drawn from the
same context distribution as their neighbours: ``--cites-only`` lifts a fixed
citation list, ``--pad-junk`` lifts its junk from a stored arm in list order, and
``--shuffle-refs`` seeds on ``SHUFFLE_SEED + i`` rather than on the clock.

The inline judge is the one ``run`` uses -- a single free Qwen3.6 seat on
NHR@FAU -- so a topped-up row carries the same inline seat as the rows beside it.
The other seats are filled afterwards by the ordinary ``rejudge`` path.

    # _gold_cites: --cites-from's own source arm was deleted on 09-11;
    # _gold_combined reproduces its citation list for 60 of 60 cases (see README)
    PYTHONPATH=analysis python -u analysis/topup_oracle_arm.py \
        --tag _gold_cites --cites-only \
        --cites-from zubaers_result/essay_writing/oracle_rag/ji2/oracle_deepseek-ai_DeepSeek-V4-Flash_gold_combined.csv \
        --model deepseek/deepseek-v4-flash-0731 --pin-deepinfra

    # _gold_pad
    PYTHONPATH=analysis python -u analysis/topup_oracle_arm.py \
        --tag _gold_pad --shuffle-refs --pad-junk 20 \
        --pad-from zubaers_result/essay_writing/rag_titled_combined_rrf_k50/ji2/rag_deepseek-ai_DeepSeek-V4-Flash.csv \
        --model deepseek/deepseek-v4-flash-0731 --pin-deepinfra

``--dry-run`` reports the gaps and the context it would build, without calling
anything.
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath('..'))

import oracle_experiment as oe

#: the validated OpenRouter routing for this model -- see endpoints.yaml
DEEPINFRA = {'order': ['DeepInfra'], 'allow_fallbacks': False,
             'data_collection': 'deny'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tag', required=True, help='e.g. _gold_cites')
    p.add_argument('--model', required=True,
                   help='the endpoint to generate with; may differ from the '
                        'name stored in the file (kept as-is)')
    p.add_argument('--arm-file', help='defaults to the one --tag names under oe.D')
    p.add_argument('--judge', default='Qwen/Qwen3.6-35B-A3B-FP8',
                   help="run's inline seat; free on NHR@FAU")
    p.add_argument('--max-passages', type=int, default=120)
    p.add_argument('--gold-set', default='gold')
    p.add_argument('--include-uncovered', action='store_true')
    p.add_argument('--shuffle-refs', action='store_true')
    p.add_argument('--cites-only', action='store_true')
    p.add_argument('--cites-from')
    p.add_argument('--pad-junk', type=int, default=0)
    p.add_argument('--pad-from')
    p.add_argument('--qa-prompt', default='qa1')
    p.add_argument('--judge-instruction', default='ji2')
    p.add_argument('--kb', default=oe.KB)
    p.add_argument('--concurrency', type=int, default=4)
    p.add_argument('--pin-deepinfra', action='store_true',
                   help='pin OpenRouter routing; ignored for a non-OR endpoint')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()

    path = a.arm_file
    if path is None:
        hits = [f for f in os.listdir(oe.D) if f.endswith(f'{a.tag}.csv')]
        if len(hits) != 1:
            raise SystemExit(f'--tag {a.tag} matches {len(hits)} file(s) in {oe.D}: {hits}')
        path = os.path.join(oe.D, hits[0])

    src = pd.read_csv(path).sort_values('index').reset_index(drop=True)
    # the same three shapes ``Judge._missing`` handles: nan, 'nan', ''
    blank = src.answer.isna() | src.answer.astype(str).str.strip().isin(('', 'nan'))
    gaps = [int(i) for i in src['index'][blank]]
    print(f'{os.path.basename(path)}: {len(src)} rows, {len(gaps)} blank '
          f'-> cases {gaps[:4]}{"..." if len(gaps) > 4 else ""}'
          f'{gaps[-1:] if len(gaps) > 4 else []}', flush=True)
    if not gaps:
        print('nothing to do')
        return

    oe.api_keys()
    facts, sols = oe.cases()
    if len(facts) != len(src):
        raise SystemExit(f'{len(facts)} cases but {len(src)} rows; refusing to '
                         'align a top-up against a different case set')
    gold, ranked = oe.gold_sets(sols, a.gold_set)
    if a.shuffle_refs:
        binding = [i for i, g in gold.items() if len(g) > a.max_passages]
        if binding:
            raise SystemExit(f'--shuffle-refs with --max-passages {a.max_passages}: '
                             f'the cap binds on {len(binding)} case(s)')
        ranked = oe.shuffle_ranked(ranked)
        print(f'reference order shuffled per case, seed {oe.SHUFFLE_SEED}', flush=True)

    ret = oe.OracleRetriever(facts, gold, kb=a.kb, max_passages=a.max_passages,
                             ranked=ranked, include_uncovered=a.include_uncovered,
                             cites_only=a.cites_only, cites_from=a.cites_from,
                             pad_junk=a.pad_junk, pad_from=a.pad_from)

    # The retriever is built over all 81 cases because cites_from/pad_from and
    # the shuffle seed are keyed on the case index, so it must see the same
    # indexing the full run did; only the gaps are then generated.
    sub_facts = [facts[i] for i in gaps]
    sub_sols = [sols[i] for i in gaps]

    if a.dry_run:
        ctx = ret.predict(sub_facts[:1])[0]
        print(f'\nexample context for case {gaps[0]} -- {len(ctx)} passages:')
        for r in ctx[:8]:
            print(f'  {r["law_book"]} {r["paragraph"]}  ({len(r["text"])} chars)')
        if len(ctx) > 8:
            print(f'  ... {len(ctx) - 8} more')
        print('\ndry run: no API calls made')
        return

    from src import qa, scoring, prompts, evaluate
    qa_plain, qa_rag = prompts.QA_PROMPTS_BY_NAME[a.qa_prompt]
    instruction = prompts.JUDGE_INSTRUCTIONS_BY_NAME[a.judge_instruction]
    judge = scoring.JudgeEnsemble(
        [scoring.Judge(model=a.judge, prompt=prompts.build_judge_user(instruction))],
        verbose=True)
    llm_kw = {'max_concurrency': a.concurrency, 'timeout': oe.GEN_TIMEOUT}
    if a.pin_deepinfra:
        llm_kw['provider'] = DEEPINFRA
    gen = qa.AnswerGenerator(retriever=ret, model=a.model, prompt=qa_rag, **llm_kw)
    print(f'generating {len(gaps)} essay(s) with {gen.model} on '
          f'{gen.inference_endpoint}'
          + (f' via {DEEPINFRA["order"]}' if a.pin_deepinfra else ''), flush=True)

    # keep the file's own model label: the endpoint that serves the weights is
    # not the identity of the arm
    stored_model = str(src.model.dropna().iloc[0]) if src.model.notna().any() else a.model
    df = evaluate.evaluate_model(gen, sub_facts, sub_sols, judge=judge,
                                 verbose=True, model_name=stored_model)
    df['index'] = gaps

    landed = df.answer.notna() & df.answer.astype(str).str.strip().ne('')
    print(f'\n{int(landed.sum())}/{len(gaps)} generated', flush=True)
    if not landed.any():
        raise SystemExit('every top-up row came back empty; nothing written')

    merged = src.set_index('index')
    for col in df.columns:
        if col == 'index':
            continue
        if col not in merged.columns:
            merged[col] = pd.NA
        merged.loc[df['index'][landed], col] = df.loc[landed.values, col].values
    merged = merged.reset_index()
    assert len(merged) == len(src)

    backup = path + '.pre_topup'
    if not os.path.exists(backup):
        src.to_csv(backup, index=False)
        print(f'backed up to {os.path.basename(backup)}', flush=True)
    merged.to_csv(path + '.tmp', index=False)
    os.replace(path + '.tmp', path)
    still = merged.answer.isna().sum()
    print(f'wrote {os.path.basename(path)}: {len(merged) - still}/{len(merged)} '
          f'essays present ({still} still blank)')


if __name__ == '__main__':
    main()
