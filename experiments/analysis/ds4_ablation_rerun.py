"""DS4-ablation-rerun: the DeepSeek oracle ablation, regenerated on OpenRouter.

The full account -- symptom, what was ruled out and how, the plan and its
acceptance criteria -- is in DS4-ablation-rerun.md at the repo root. In short:
the five DeepSeek-V4-Flash oracle-ablation arms generated on NHR@FAU in the run
that ended with the 2026-09-12 outage score 6-9 points above every earlier DS4
generation of the same conditions, while `_gold_top10` (never regenerated) and a
Gemma-4 control regenerated today both reproduce their published cells. The
FAU DS4 deployment has been down since, so the window cannot be probed.

This regenerates those five arms through OpenRouter pinned to DeepInfra, with
inputs byte-identical to the FAU window arms (``verify`` checks every stored
prompt before anything is spent), into their own directory under canonical
filenames, then judges them on the free3 seats. Swapping them into the tables is
a separate, deliberate step, taken only if the acceptance criteria hold.

    PYTHONPATH=analysis python -u analysis/ds4_ablation_rerun.py verify
    PYTHONPATH=analysis python -u analysis/ds4_ablation_rerun.py generate --concurrency 40
    PYTHONPATH=analysis python -u analysis/ds4_ablation_rerun.py judge _gold_combined --wait
    PYTHONPATH=analysis python -u analysis/ds4_ablation_rerun.py status

Generation is resumable per essay: every finished essay is appended to a
``.jsonl.part`` checkpoint, and re-running ``generate`` skips what is there.
There is no chunk barrier, so one 40-minute essay holds up nothing else.

``--backend gwdg`` runs the same arms, prompts and judges into DS4-ablation-gwdg
on the GWDG academic cloud (ENDPOINT_AC / API_KEY_AC), at reasoning effort
``medium``. It tests whether the reasoning budget explains the oracle gap:

    PYTHONPATH=analysis python -u analysis/ds4_ablation_rerun.py --backend gwdg generate --tags _gold_combined --concurrency 10
    PYTHONPATH=analysis python -u analysis/ds4_ablation_rerun.py --backend gwdg judge _gold_combined --wait
"""
import argparse
import datetime
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath('..'))

import oracle_experiment as oe

RUN_ID = 'DS4-ablation-rerun'
OUT = os.path.join(os.path.dirname(oe.D), RUN_ID)
META = os.path.join(OUT, 'run.json')

#: the endpoint that serves the weights, and the identity the arms are stored under
ENDPOINT_MODEL = 'deepseek/deepseek-v4-flash-0731'
STORED = 'deepseek-ai/DeepSeek-V4-Flash'
SLUG = STORED.replace('/', '_')
ENDPOINT_LABEL = 'openrouter:DeepInfra'
#: see endpoints.yaml: of four providers on this slug one is biased +13.75 and one refuses
PROVIDER = {'order': ['DeepInfra'], 'allow_fallbacks': False, 'data_collection': 'deny'}

#: Where the arms are generated; the judges are the same for both. NHR@FAU's DS4
#: reasoned 10.7k tokens (median) per no-RAG essay in August at `high`, DeepInfra
#: 44.6k on the same prompts -- fewer on 81 of 81 cases -- and the FAU window arms
#: since its restart ~41k. GWDG reasons 9.6-10.7k at `low` and `medium` and
#: 13-14k at `high` on the same prompts (two cases each, 2026-09-13), so `gwdg`
#: regenerates at the August budget. GWDG counts every request against 30/min,
#: 200/h, 1000/day, 3000/month (X-RateLimit-* headers); it streams usage.
#: `openrouter-low` (DS4-ablation-low) is DeepInfra at `low`, the ladder's regime:
#: DeepSeek's encoding realises effort as a prompt prefix (low: none, high:
#: "Absolute maximum...", max: "Beyond maximum..."), DeepInfra applies it for
#: `high`, and August FAU sent `high` without it. On no-RAG case 46 DeepInfra
#: gives 14.1k tokens at `low`, 11.3k at `medium`, 44.1k at `high`.
BACKENDS = {
    'openrouter': {'run_id': 'DS4-ablation-rerun', 'model': 'deepseek/deepseek-v4-flash-0731',
                   'label': 'openrouter:DeepInfra', 'provider': PROVIDER, 'effort': 'high',
                   'endpoint_var': None, 'token_var': None, 'stream': None},
    'openrouter-low': {'run_id': 'DS4-ablation-low', 'model': 'deepseek/deepseek-v4-flash-0731',
                       'label': 'openrouter:DeepInfra:low', 'provider': PROVIDER, 'effort': 'low',
                       'endpoint_var': None, 'token_var': None, 'stream': None},
    'gwdg': {'run_id': 'DS4-ablation-gwdg', 'model': 'deepseek-v4-flash-0731',
             'label': 'gwdg:medium', 'provider': None, 'effort': 'medium',
             'endpoint_var': 'ENDPOINT_AC', 'token_var': 'API_KEY_AC', 'stream': True},
}
BACKEND = BACKENDS['openrouter']


def use_backend(name):
    global BACKEND, RUN_ID, OUT, META, ENDPOINT_MODEL, ENDPOINT_LABEL
    BACKEND = BACKENDS[name]
    RUN_ID = BACKEND['run_id']
    OUT = os.path.join(os.path.dirname(oe.D), RUN_ID)
    META = os.path.join(OUT, 'run.json')
    ENDPOINT_MODEL, ENDPOINT_LABEL = BACKEND['model'], BACKEND['label']


MAX_PASSAGES = 120
PAD_FROM = ('zubaers_result/essay_writing/rag_titled_combined_rrf_k50/ji2/'
            'rag_deepseek-ai_DeepSeek-V4-Flash.csv')
#: `_gold_cites` was built --cites-from `_gold_now.csv`, deleted 2026-09-11;
#: `_gold_combined` reproduces its citation list for 60 of 60 cases (README)
CITES_FROM = f'{oe.D}/oracle_{SLUG}_gold_combined.csv'

#: the flags run_ds_ablation_fau.sh gave each arm. `_gold_combined` first: it is
#: the acceptance cell, so it should land earliest.
CONDITIONS = {
    '_gold_combined': {},
    '_gold_top10': {'max_passages': 10},
    '_gold_unc': {'include_uncovered': True},
    '_gold_shuf': {'shuffle_refs': True},
    '_gold_pad': {'shuffle_refs': True, 'pad_junk': 20, 'pad_from': PAD_FROM},
    '_gold_cites': {'cites_only': True, 'cites_from': CITES_FROM},
}
#: free3, in the order that banks the free seats first
SEATS = [('openai/gpt-oss-120b', None),
         ('Qwen/Qwen3.6-35B-A3B-FP8', None),
         ('deepseek/deepseek-v4-flash-0731', PROVIDER)]
DRAFT = {'_gold_combined': 4.13, '_gold_top10': 6.98, '_gold_unc': 3.64, '_gold_shuf': 2.47,
         '_gold_pad': 0.19, '_gold_cites': 0.19}
#: the phrases the unpinned 2026-09-11 OpenRouter run refused with. Matched only
#: in an essay's opening: "liegt nicht vor" is ordinary legal reasoning ("eine
#: verfassungsrechtliche Streitigkeit liegt nicht vor") and flagged seven full
#: 13-19k-char essays as refusals on the first pass. A refusal is short and says
#: so up front.
REFUSAL = ('kann nicht erstellt werden', 'nicht enthalten', 'liegt nicht vor',
           'fehlt der sachverhalt', 'nicht in der erwarteten form')
REFUSAL_WINDOW = 400


def looks_refused(answer):
    a = str(answer or '')
    return len(a) < 3000 and any(p in a[:REFUSAL_WINDOW].lower() for p in REFUSAL)
ROUNDS = 3
#: an arm with more holes than this after ROUNDS is left unfinalised, so a
#: transient outage cannot hand a half-empty arm to the judges
MAX_HOLES = 5


def final_path(tag):
    return f'{OUT}/oracle_{SLUG}{tag}.csv'


def part_path(tag):
    return f'{OUT}/oracle_{SLUG}{tag}.jsonl.part'


def fau_path(tag):
    return f'{oe.D}/oracle_{SLUG}{tag}.csv'


def now():
    return datetime.datetime.now().isoformat(timespec='seconds')


def log(msg):
    print(f'{time.strftime("%F %T")}  {msg}', flush=True)


def meta(**kw):
    os.makedirs(OUT, exist_ok=True)
    m = json.load(open(META)) if os.path.exists(META) else {'run_id': RUN_ID}
    m.update(kw)
    json.dump(m, open(META + '.tmp', 'w'), indent=2, ensure_ascii=False)
    os.replace(META + '.tmp', META)
    return m


def make_gen():
    from src import qa, prompts
    _, qa_rag = prompts.QA_PROMPTS_BY_NAME['qa1']
    b = BACKEND
    kw = {'provider': b['provider']} if b['provider'] else {}
    if b['endpoint_var']:
        kw.update(inference_endpoint=os.environ[b['endpoint_var']], token_var=b['token_var'])
    return qa.AnswerGenerator(model=ENDPOINT_MODEL, prompt=qa_rag, max_concurrency=1,
                              timeout=oe.GEN_TIMEOUT, reasoning_effort=b['effort'],
                              stream=b['stream'], **kw)


def build_messages(tag, gen, facts, sols):
    """The exact messages ``run`` sent for this arm, for all 81 cases."""
    c = CONDITIONS[tag]
    gold, ranked = oe.gold_sets(sols, 'gold')
    if c.get('shuffle_refs'):
        if any(len(g) > MAX_PASSAGES for g in gold.values()):
            raise SystemExit(f'{tag}: the passage cap binds, shuffle would change membership')
        ranked = oe.shuffle_ranked(ranked)
    ret = oe.OracleRetriever(facts, gold, kb=oe.KB, max_passages=c.get('max_passages', MAX_PASSAGES),
                             ranked=ranked, verbose=False,
                             include_uncovered=c.get('include_uncovered', False),
                             cites_only=c.get('cites_only', False),
                             cites_from=c.get('cites_from'),
                             pad_junk=c.get('pad_junk', 0), pad_from=c.get('pad_from'))
    ctxs = ret.predict(facts)
    # exactly what AnswerGenerator.predict does before build_messages
    return [gen.build_messages(f'<context>\n{gen.format_context(x)}\n</context>', f)
            for x, f in zip(ctxs, facts)]


def prepare(tags, gen, facts, sols):
    """Build every arm's messages and check them against the FAU window arm."""
    msgs, report, ok = {}, {}, True
    for tag in tags:
        msgs[tag] = build_messages(tag, gen, facts, sols)
        src = pd.read_csv(fau_path(tag), usecols=['index', 'qa_prompt']).sort_values('index')
        live = src[src.qa_prompt.notna()]
        same = sum(str(msgs[tag][int(i)]) == str(q) for i, q in zip(live['index'], live.qa_prompt))
        report[tag] = {'identical': int(same), 'compared': int(len(live))}
        log(f'verify {tag:<15} {same}/{len(live)} stored prompts rebuilt byte-identically')
        ok &= same == len(live)
    meta(verify=report, verified_at=now(), verify_ok=bool(ok))
    return msgs, ok


def load_part(tag):
    rows = {}
    if os.path.exists(part_path(tag)):
        for line in open(part_path(tag)):
            if line.strip():
                r = json.loads(line)
                if r.get('answer') and str(r['answer']).strip():
                    rows[int(r['index'])] = r
    return rows


def write_final(tag, rows, sols):
    from src import scoring
    recs = [rows.get(i, {'index': i, 'model': STORED, 'answer': None}) for i in range(len(sols))]
    df = pd.DataFrame(recs).sort_values('index').reset_index(drop=True)
    df['legal_ref_sim'] = [scoring.legal_ref_similarity(s, a)
                           if isinstance(a, str) and a.strip() else np.nan
                           for a, s in zip(df.answer, sols)]
    df['run_id'], df['endpoint'] = RUN_ID, ENDPOINT_LABEL
    df.to_csv(final_path(tag) + '.tmp', index=False)
    os.replace(final_path(tag) + '.tmp', final_path(tag))


def generate(concurrency, skip_verify=False, only=None):
    oe.api_keys()
    facts, sols = oe.cases()
    n = len(facts)
    gen = make_gen()
    tags = [t for t in (only or CONDITIONS) if not os.path.exists(final_path(t))]
    if not tags:
        log('every arm already generated'); return
    msgs, ok = prepare(tags, gen, facts, sols)
    if not ok and not skip_verify:
        raise SystemExit('inputs differ from the FAU window arms; refusing to generate')
    meta(endpoint=ENDPOINT_LABEL, endpoint_model=ENDPOINT_MODEL, stored_model=STORED,
         provider=BACKEND['provider'], reasoning_effort=BACKEND['effort'],
         max_passages=MAX_PASSAGES, conditions={t: CONDITIONS[t] for t in tags},
         generation_started=meta().get('generation_started', now()),
         concurrency=concurrency)
    done = {t: load_part(t) for t in tags}
    lock = threading.Lock()

    def work(job):
        t, i = job
        t0 = time.time()
        res = gen._call_with_retry(msgs[t][i])
        ans = gen.postprocess(res['message']) if res else None
        row = {'index': i, 'model': STORED, 'answer': ans, 'qa_prompt': str(msgs[t][i]),
               'finish_reason': res.get('finish_reason') if res else None,
               'prompt_tokens': res.get('prompt_tokens') if res else None,
               'completion_tokens': res.get('completion_tokens') if res else None,
               'total_tokens': res.get('total_tokens') if res else None,
               'total_cost': res.get('total_cost') if res else None,
               'reasoning_effort': BACKEND['effort'],
               'time': round(time.time() - t0, 1), 'run_id': RUN_ID,
               'endpoint': ENDPOINT_LABEL, 'generated_at': now()}
        good = bool(ans and str(ans).strip())
        with lock:
            if good:
                with open(part_path(t), 'a') as f:
                    f.write(json.dumps(row, ensure_ascii=False) + '\n')
                done[t][i] = row
                if len(done[t]) == n:
                    write_final(t, done[t], sols)
                    log(f'### {t} complete {n}/{n} -> {final_path(t)}')
        return t, i, good, row

    for rnd in range(1, ROUNDS + 1):
        jobs = [(t, i) for t in tags for i in range(n) if i not in done[t]]
        if not jobs:
            break
        log(f'round {rnd}: {len(jobs)} generation(s) pending, concurrency {concurrency}')
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futs = [ex.submit(work, j) for j in jobs]
            for k, fut in enumerate(as_completed(futs), 1):
                try:
                    t, i, good, row = fut.result()
                except Exception as e:           # one bad worker must not end the run
                    log(f'!! worker error {type(e).__name__}: {str(e)[:200]}'); continue
                a = str(row['answer'] or '')
                flag = '  REFUSAL?' if good and looks_refused(a) else ''
                log(f'{k}/{len(jobs)} {t:<15} case {i:>2} {"ok" if good else "FAILED":<6} '
                    f'{len(a):>6} ch {row["time"]:>6}s ${row["total_cost"] or 0:.4f}{flag}'
                    f'   [{t} {len(done[t])}/{n}]')
    for t in tags:
        if os.path.exists(final_path(t)):
            continue
        holes = [i for i in range(n) if i not in done[t]]
        if len(holes) <= MAX_HOLES:
            write_final(t, done[t], sols)
            log(f'### {t} finalised with {len(holes)} blank row(s): {holes}')
        else:
            log(f'!! {t}: {len(holes)} essays still missing after {ROUNDS} rounds; '
                'not finalised -- re-run generate to resume')
    meta(generation_last_exit=now())


def arm_lock(tag):
    """Serialise the re-read/graft/write of one arm across seat processes."""
    import fcntl, tempfile
    f = open(os.path.join(tempfile.gettempdir(), f'{RUN_ID}{tag}.lock'), 'w')
    fcntl.flock(f, fcntl.LOCK_EX)
    return f


def judge(tag, wait, concurrency, only_seats=None):
    p = final_path(tag)
    while not os.path.exists(p):
        if not wait:
            raise SystemExit(f'{p} not generated yet')
        time.sleep(60)
    log(f'### judging {tag}')
    oe.api_keys()
    facts, sols = oe.cases()
    from src import scoring, prompts, evaluate
    instruction = prompts.JUDGE_INSTRUCTIONS_BY_NAME['ji2']
    for seat, prov in SEATS:
        name = seat.split('/')[-1]
        if only_seats and name not in only_seats:
            continue
        col = f'score_Judge ({name})'
        d = pd.read_csv(p).sort_values('index').reset_index(drop=True)
        if col in d.columns and d[col].notna().any():
            log(f'  {name}: already present, skipping'); continue
        kw = {'max_concurrency': concurrency, **({'provider': prov} if prov else {})}
        ens = scoring.JudgeEnsemble(
            [scoring.Judge(model=seat, prompt=prompts.build_judge_user(instruction), **kw)],
            verbose=True)
        t0 = time.time()
        out = evaluate.rejudge_model(d.answer.tolist(), facts, sols, judge=ens,
                                     model_name=STORED, verbose=True)
        add = [c for c in out.columns if c.endswith(f'Judge ({name})')]
        # seats may run as separate processes (--seats): re-read under a lock
        lock = arm_lock(tag)
        d = pd.read_csv(p).sort_values('index').reset_index(drop=True)
        for c in add:
            d[c] = out[c].values
        d.to_csv(p + '.tmp', index=False)
        os.replace(p + '.tmp', p)
        lock.close()
        s = d[col]
        log(f'  {name}: n={int(s.notna().sum())} mean {100 * s.mean():.2f} '
            f'cost ${out.judging_cost.sum():.2f} ({time.time() - t0:.0f}s)')
    log(f'### {tag} judged')


def status():
    import paper_table as pt
    m = json.load(open(META)) if os.path.exists(META) else {}
    print(f'{RUN_ID}   dir {OUT}')
    print(f'verify_ok {m.get("verify_ok")}   generation_started {m.get("generation_started")}'
          f'   concurrency {m.get("concurrency")}')
    scale = pt.SCALES['free3']
    base = np.asarray(pt.baseline(STORED, scale, None), dtype=float)
    lv = pd.read_csv('analysis/out/ladder_values.csv')
    ladder = float(lv[(lv.block == 'Essay score (0--100)') & (lv.col == 'oracle')
                      & (lv.row == 'DeepSeek-V4-Flash')].value.iloc[0])
    ref_dir = os.path.join(os.path.dirname(oe.D), BACKENDS['openrouter']['run_id'])
    print(f'\n{"arm":<15}{"gen":>7}{"$gen":>7}{"ref?":>5}{"tok":>6}  {"gpt-oss":>8}{"Qwen":>7}{"DS/DI":>7}'
          f'{"free3":>7}{"delta":>8}{"2SEM":>6}{"draft":>7}{"FAU-win":>8}{"OR-rr":>7}')
    for tag in CONDITIONS:
        fin = os.path.exists(final_path(tag))
        d = pd.read_csv(final_path(tag)) if fin else pd.DataFrame(list(load_part(tag).values()))
        ngen = int(d.answer.notna().sum()) if len(d) else 0
        cost = float(pd.to_numeric(d.get('total_cost'), errors='coerce').sum()) if len(d) else 0.0
        refs = int(sum(looks_refused(a) for a in d.answer.dropna())) if len(d) else 0
        ct = pd.to_numeric(d.get('completion_tokens'), errors='coerce') if len(d) else pd.Series(dtype=float)
        tok = f'{ct.median() / 1000:.1f}k' if ct.notna().any() else '-'
        seats = []
        for seat, _ in SEATS:
            c = f'score_Judge ({seat.split("/")[-1]})'
            seats.append(f'{100 * d[c].mean():.1f}' if fin and c in d and d[c].notna().any() else '-')
        f3 = dl = se = '-'
        if fin:
            v = pt.med(d.sort_values('index'), scale)
            if v is not None:
                v = np.asarray(v, dtype=float)
                k = min(len(v), len(base))
                dd = pd.Series(v[:k] - base[:k]).dropna()
                f3, dl = f'{np.nanmean(v):.2f}', f'{dd.mean():+.2f}'
                se = f'{2 * dd.std(ddof=1) / np.sqrt(len(dd)):.2f}'
        ref = {}
        for key, path in (('fw', fau_path(tag)), ('rr', f'{ref_dir}/oracle_{SLUG}{tag}.csv')):
            ref[key] = '-'
            if os.path.exists(path):
                w = pt.med(pd.read_csv(path).sort_values('index'), scale)
                if w is not None:
                    w = np.asarray(w, dtype=float); k = min(len(w), len(base))
                    ref[key] = f'{pd.Series(w[:k] - base[:k]).dropna().mean():+.2f}'
        print(f'{tag:<15}{ngen:>4}/81{cost:>7.2f}{refs:>5}{tok:>6}  {seats[0]:>8}{seats[1]:>7}{seats[2]:>7}'
              f'{f3:>7}{dl:>8}{se:>6}{DRAFT[tag]:>+7.2f}{ref["fw"]:>8}{ref["rr"]:>7}')
    print(f'\nacceptance (b): _gold_combined free3 ~ ladder {ladder:.2f}, '
          f'delta ~ +{ladder - np.nanmean(base):.2f} within 2SEM')
    nr = f'{oe.D}/norag_regime_{SLUG}.csv'
    nr = nr if os.path.exists(nr) else nr + '.part'
    if os.path.exists(nr):
        d = pd.read_csv(nr).sort_values('index')
        c = 'score_Judge (gpt-oss-120b)'
        row = pt.main_norag()
        row = row[row.model == pt.MAIN_NORAG_ID.get(STORED, STORED)].sort_values('index')
        a = 100 * d[c].to_numpy(dtype=float); b = 100 * row[c].to_numpy(dtype=float)[:len(a)]
        dd = pd.Series(a - b).dropna()
        print(f'acceptance (a): fresh OR no-RAG vs banked FAU no-RAG, gpt-oss seat, '
              f'n={len(dd)}: shift {dd.mean():+.2f}  2SEM {2 * dd.std(ddof=1) / np.sqrt(len(dd)):.2f}'
              f'{"   (partial)" if nr.endswith(".part") else ""}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--backend', choices=list(BACKENDS), default='openrouter')
    sub = ap.add_subparsers(dest='cmd', required=True)
    v = sub.add_parser('verify')
    v.add_argument('--tags', nargs='*', default=list(CONDITIONS), choices=list(CONDITIONS))
    g = sub.add_parser('generate')
    g.add_argument('--concurrency', type=int, default=40)
    g.add_argument('--skip-verify', action='store_true')
    g.add_argument('--tags', nargs='*', choices=list(CONDITIONS), help='default: every arm')
    j = sub.add_parser('judge')
    j.add_argument('tag', choices=list(CONDITIONS))
    j.add_argument('--wait', action='store_true', help='poll until the arm is generated')
    j.add_argument('--concurrency', type=int, default=12)
    j.add_argument('--seats', nargs='*', choices=[s.split('/')[-1] for s, _ in SEATS],
                   help='judge only these seats, e.g. to run a slow seat in its own process')
    sub.add_parser('status')
    a = ap.parse_args()
    use_backend(a.backend)
    if a.cmd == 'verify':
        oe.api_keys()
        facts, sols = oe.cases()
        _, ok = prepare(a.tags, make_gen(), facts, sols)
        raise SystemExit(0 if ok else 1)
    if a.cmd == 'generate':
        generate(a.concurrency, a.skip_verify, a.tags)
    elif a.cmd == 'judge':
        judge(a.tag, a.wait, a.concurrency, a.seats)
    else:
        status()


if __name__ == '__main__':
    main()
