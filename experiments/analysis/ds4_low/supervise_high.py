"""Supervisor for the DeepSeek {\\tiny high} arms on GWDG: idempotent, run from cron.

Every tick (``python analysis/ds4_low/supervise_high.py tick``, from experiments/):

1. generation -- keeps at most ``GEN_SLOTS`` arms generating on GWDG, in order: the
   nine missing RAG cells, then the ls_text rerun, then the no-RAG rerun. A process
   that died is simply relaunched; corpus_rag_run and run_model both resume.
2. judging -- for every generated arm, each seat (gpt-oss-120b and Qwen3.6-35B-A3B-FP8
   on NHR@FAU, DeepSeek at low on DeepInfra) is its own side file; a seat that is
   missing or has an unscored live row is relaunched unless already running.
3. merge -- once all three seats are complete, their columns are written into the
   arm file.
4. integration -- once every arm is merged, ``integrate_high_gwdg.py`` (if present)
   runs once, then git commit + push.

State (launch counts, done markers) lives in ``logs/high_gwdg_state.json``; a
human-readable summary in ``logs/high_gwdg_status.txt``.
"""
import json, os, subprocess, sys, time
import pandas as pd

EXP = os.path.join(os.environ.get('GPBAM_ROOT') or os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'experiments')
PY = os.environ.get('GPBAM_PYTHON', sys.executable)
E = 'zubaers_result/essay_writing'
STATE, STATUS = 'logs/high_gwdg_state.json', 'logs/high_gwdg_status.txt'
GEN_SLOTS = 2
MAX_LAUNCH = {'gen': 25, 'judge': 40}
PROC_CAP = {'gpt-oss': 4, 'qwen': 4, 'deepseek-low': 4}
ARM_FILE = 'rag_deepseek-ai_DeepSeek-V4-Flash.csv'
RAG = [('combined', 'rrf', 50, 'rag_titled_combined_rrf_k50'),
       ('combined', 'rrf', 10, 'rag_titled_combined_rrf'),
       ('combined', 'rerank', 10, 'rag_titled_combined_rerank'),
       ('combined', 'cite_only', 10, 'rag_titled_combined_cite_only'),
       ('combined', 'cite_rrf', 10, 'rag_titled_combined_cite_rrf'),
       ('combined', 'cite_rrf', 50, 'rag_titled_combined_cite_rrf_k50'),
       ('combined', 'cite_ls_rrf', 50, 'rag_titled_combined_cite_ls_rrf_k50'),
       ('combined', 'roundrobin', 10, 'rag_titled_combined'),
       ('federal', 'roundrobin', 10, 'rag_titled_federal')]
LATE = [('combined', 'ls_text', 10, 'rag_titled_combined_ls_text')]
NORAG_D = f'{E}/without_rag_high_gwdg'
SEATS = {'gpt-oss': ('judge_gpt-oss-120b.csv', 'score_Judge (gpt-oss-120b)'),
         'qwen': ('judge_Qwen3.6-35B-A3B-FP8.csv', 'score_Judge (Qwen3.6-35B-A3B-FP8)'),
         'deepseek-low': ('judge_effort_low_deepseek.csv', 'score_Judge (deepseek-v4-flash-0731)')}


def arms():
    out = []
    for c, p, k, base in RAG + LATE:
        d = f'{E}/{base}/DS4-high-gwdg'
        out.append(dict(tag=f'{c}_{p}_k{k}', kind='rag', corpus=c, pipeline=p, k=k, dir=d,
                        arm=f'{d}/{ARM_FILE}', late=(c, p, k, base) in LATE))
    out.append(dict(tag='norag', kind='norag', dir=NORAG_D, arm=f'{NORAG_D}/norag_deepseek-ai_DeepSeek-V4-Flash.csv', late=True))
    return out


def procs():
    """Command lines of running python processes only.

    Shells are excluded: a shell whose command line merely *mentions* a script
    (an editor session, a heredoc) must not count as that script running.
    """
    out = []
    for line in subprocess.run(['ps', '-eo', 'args'], capture_output=True, text=True).stdout.splitlines():
        exe = line.split(' ', 1)[0]
        if exe.endswith(('/python', '/python3')) or exe in ('python', 'python3'):
            out.append(line)
    return out


def running(ps, *needles):
    return any(all(n in line for n in needles) for line in ps)


def launch(cmd, log):
    env = dict(os.environ, PYTHONPATH='analysis')
    with open(log, 'a') as fh:
        fh.write(f'\n### {time.strftime("%F %T")} supervisor launch: {" ".join(cmd)}\n')
        fh.flush()
        subprocess.Popen(cmd, cwd=EXP, env=env, stdout=fh, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, start_new_session=True)


def live_rows(arm):
    d = pd.read_csv(arm, usecols=['index', 'answer'], low_memory=False)
    return d, [i for i, a in zip(d['index'], d['answer']) if isinstance(a, str) and a.strip()]


def seat_state(a, live):
    out = {}
    for seat, (fname, col) in SEATS.items():
        p = f'{a["dir"]}/{fname}'
        if not os.path.exists(p):
            out[seat] = (0, False); continue
        s = pd.read_csv(p, low_memory=False)
        if col not in s or 'index' not in s:
            out[seat] = (0, False); continue
        scored = set(s.loc[s[col].notna(), 'index'].astype(int))
        out[seat] = (len(scored & set(live)), set(live) <= scored)
    return out


def merge(a):
    d = pd.read_csv(a['arm'], low_memory=False).sort_values('index').reset_index(drop=True)
    for seat, (fname, col) in SEATS.items():
        s = pd.read_csv(f'{a["dir"]}/{fname}', low_memory=False).sort_values('index').reset_index(drop=True)
        assert (s['index'].values == d['index'].values).all(), (a['tag'], seat)
        for c in [c for c in s.columns if c.endswith(col[len('score_'):])]:
            d[c] = s[c].values
    d['deepseek_judge'] = 'openrouter:DeepInfra:low'
    d.to_csv(a['arm'] + '.tmp', index=False)
    os.replace(a['arm'] + '.tmp', a['arm'])


def tick():
    os.chdir(EXP)
    st = json.load(open(STATE)) if os.path.exists(STATE) else {'launch': {}, 'merged': [], 'integrated': False}
    ps = procs()
    lines = [f'DS4 {{\\tiny high}} on GWDG -- supervisor tick {time.strftime("%F %T")}']
    gens = sum('corpus_rag_run.py generate' in l and '--endpoint gwdg' in l for l in ps) + \
        sum('norag_high_gwdg.py generate' in l for l in ps)
    all_arms = arms()
    # an arm that keeps failing before it saves anything does not hold up the final reruns
    def settled(a):
        failing = st['launch'].get(f'gen:{a["tag"]}', 0) >= 3 and not os.path.exists(a['arm'] + '.part')
        return os.path.exists(a['arm']) or failing
    rag_done = all(settled(a) for a in all_arms if a['kind'] == 'rag' and not a['late'])
    late_done = all(os.path.exists(a['arm']) for a in all_arms if a['kind'] == 'rag' and a['late'])
    counts = {seat: sum(f'--seat {seat}' in l and 'judge_low_arms.py' in l for l in ps) for seat in SEATS}

    # -- federal store rebuild (owner, 2026-09-14): once the ls_text and no-RAG reruns are
    #    generated and nothing is generating, rebuild my_knowledge_base_titled from the combined
    #    store, then let the federal round-robin arm be launched afresh
    FED_TAG, FED_MARK = 'federal_roundrobin_k10', 'logs/high_gwdg_federal_rebuilt.json'
    fed = next(a for a in all_arms if a['tag'] == FED_TAG)
    late_generated = all(os.path.exists(a['arm']) for a in all_arms if a['late'])
    rebuild_running = any('rebuild_federal_from_combined.py' in l for l in ps)
    if not os.path.exists(fed['arm']) and not os.path.exists(FED_MARK):
        if rebuild_running:
            lines.append('federal store: REBUILDING from the combined store')
        elif late_generated and gens == 0 and st.get('rebuild_attempts', 0) < 2:
            launch([PY, '-u', 'analysis/ds4_low/rebuild_federal_from_combined.py'], 'logs/high_gwdg_federal_rebuild.log')
            st['rebuild_attempts'] = st.get('rebuild_attempts', 0) + 1
            lines.append(f'federal store: rebuild LAUNCHED (attempt {st["rebuild_attempts"]})')
        elif st.get('rebuild_attempts', 0) >= 2:
            lines.append('federal store: rebuild FAILED twice -- see logs/high_gwdg_federal_rebuild.log; integrating without it')
        else:
            lines.append('federal store: rebuild waits for the ls_text and no-RAG generations')
    if os.path.exists(FED_MARK) and not st.get('fed_reset'):
        st['launch'][f'gen:{FED_TAG}'] = 0
        st['fed_reset'] = True
        lines.append('federal store rebuilt; federal round-robin arm re-armed')

    for a in all_arms:
        tag, key = a['tag'], f'gen:{a["tag"]}'
        if not os.path.exists(a['arm']):
            if a['kind'] == 'rag':
                is_running = running(ps, 'corpus_rag_run.py generate', f'--out-dir {a["dir"]}')
                part = a['arm'] + '.part'
                n = len(pd.read_csv(part, usecols=['index'])) if os.path.exists(part) else 0
            else:
                is_running = running(ps, 'norag_high_gwdg.py generate')
                j = f'{a["dir"]}/ds4_gwdg_high_essays.jsonl'
                n = sum(1 for _ in open(j)) if os.path.exists(j) else 0
            # ls_text and no-RAG both come last, side by side once the nine RAG cells are generated
            allowed = (not a['late']) or rag_done
            if is_running:
                lines.append(f'{tag:<26} generating  {n}/81 saved')
            elif a['kind'] == 'rag' and n == 0 and st['launch'].get(key, 0) >= 3:
                lines.append(f'{tag:<26} FAILING: {st["launch"][key]} launches, nothing saved -- not relaunched, needs a human')
            elif not allowed:
                lines.append(f'{tag:<26} waiting (runs after the earlier arms)')
            elif gens >= GEN_SLOTS:
                lines.append(f'{tag:<26} queued      {n}/81 saved')
            elif st['launch'].get(key, 0) >= MAX_LAUNCH['gen']:
                lines.append(f'{tag:<26} STUCK: generation launched {st["launch"][key]} times -- needs a human')
            else:
                if a['kind'] == 'rag':
                    cmd = [PY, '-u', 'analysis/corpus_rag_run.py', 'generate', '--pipeline', a['pipeline'],
                           '--top-k', str(a['k']), '--corpora', a['corpus'], '--out-dir', a['dir'],
                           '--models', 'deepseek-v4-flash-0731', '--store-as', 'deepseek-ai/DeepSeek-V4-Flash',
                           '--endpoint', 'gwdg', '--effort', 'low', '--effort-prefix', 'high', '--concurrency', '16']
                else:
                    cmd = [PY, '-u', 'analysis/ds4_low/norag_high_gwdg.py', 'generate']
                launch(cmd, f'logs/high_gwdg_arm_{tag}.log')
                st['launch'][key] = st['launch'].get(key, 0) + 1
                gens += 1
                lines.append(f'{tag:<26} LAUNCHED generation (launch #{st["launch"][key]}), {n}/81 saved')
            continue

        d, live = live_rows(a['arm'])
        blanks = len(d) - len(live)
        seats = seat_state(a, live)
        parts = []
        for seat, (n, complete) in seats.items():
            fname = SEATS[seat][0]
            side = f'{a["dir"]}/{fname}'
            jkey = f'judge:{tag}:{seat}'
            if complete:
                parts.append(f'{seat} {n}/{len(live)} done'); continue
            if running(ps, 'judge_low_arms.py', f'--seat {seat}', side):
                parts.append(f'{seat} {n}/{len(live)} running'); continue
            if st['launch'].get(jkey, 0) >= MAX_LAUNCH['judge']:
                parts.append(f'{seat} STUCK'); continue
            if counts[seat] >= PROC_CAP[seat]:
                parts.append(f'{seat} {n}/{len(live)} queued'); continue
            cmd = [PY, '-u', 'analysis/ds4_low/judge_low_arms.py', '--seat', seat]
            if seat == 'deepseek-low':
                cmd += ['--endpoint', 'deepinfra']
            launch(cmd + [a['arm'], side], f'logs/high_gwdg_judge_{seat}_{tag}.log')
            st['launch'][jkey] = st['launch'].get(jkey, 0) + 1
            counts[seat] += 1
            parts.append(f'{seat} {n}/{len(live)} LAUNCHED')
        if all(c for _, c in seats.values()):
            if tag not in st['merged']:
                merge(a); st['merged'].append(tag)
            parts.append('merged')
        lines.append(f'{tag:<26} generated {len(live)}/{len(d)}' + (f' BLANKS {blanks}' if blanks else '') + ' | ' + ', '.join(parts))

    failing_tags = [a['tag'] for a in all_arms if not os.path.exists(a['arm'])
                    and st['launch'].get(f"gen:{a['tag']}", 0) >= 3 and not os.path.exists(a['arm'] + '.part')
                    and a['kind'] == 'rag']
    # while the federal store can still be rebuilt, do not integrate around the federal arm
    if FED_TAG in failing_tags and not os.path.exists(FED_MARK) and st.get('rebuild_attempts', 0) < 2:
        failing_tags = [t for t in failing_tags if t != FED_TAG] + ['__pending_rebuild__']
    if len(st['merged']) + len([t for t in failing_tags if t != '__pending_rebuild__']) == len(all_arms) and not st['integrated']:
        if failing_tags:
            lines.append(f'integrating without failing arm(s): {failing_tags}')
        if os.path.exists('analysis/ds4_low/integrate_high_gwdg.py'):
            r = subprocess.run([PY, 'analysis/ds4_low/integrate_high_gwdg.py'], cwd=EXP, capture_output=True, text=True,
                               env=dict(os.environ, PYTHONPATH='analysis'))
            open('logs/high_gwdg_integrate.log', 'a').write(r.stdout + r.stderr)
            st['integrated'] = r.returncode == 0
            lines.append(f'INTEGRATION {"done" if r.returncode == 0 else "FAILED (see logs/high_gwdg_integrate.log)"}')
            if st['integrated']:
                # the job is finished: take this supervisor out of the crontab
                cur = subprocess.run(['crontab', '-l'], capture_output=True, text=True).stdout.splitlines()
                keep = [l for l in cur if 'supervise_high.py tick' not in l and 'DS4-high-gwdg supervisor' not in l]
                subprocess.run(['crontab', '-'], input='\n'.join(keep) + ('\n' if keep else ''), text=True)
                lines.append('cron entries removed')
        else:
            lines.append('ALL ARMS JUDGED -- integration script not present yet')
    elif st['integrated']:
        lines.append('INTEGRATED')
    json.dump(st, open(STATE, 'w'), indent=1)
    open(STATUS, 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    tick()
