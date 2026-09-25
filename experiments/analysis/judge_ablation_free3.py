"""Fill the two missing free3 seats on each DSv4 ablation arm as it goes idle.

The ablation table reads ``pt.SCALES['free3']`` (all three seats), but
``oracle_experiment.py run`` only judges inline with Qwen3.6. The supervisor
fires ``rejudge --judges free3`` after *every* arm finishes, which puts ~970
judge calls -- roughly a day of the DeepSeek seat -- entirely after the last
generation. Judging each arm the moment it lands instead overlaps that with the
arms still generating.

``_gold_combined`` is deliberately skipped: its cell is already published from
the pre-deletion run, so re-judging it only buys a run-to-run variance check.

rejudge() has no per-file selector (``--limit`` slices an alphabetical list, in
which _gold_combined sorts first), so ``targets()`` is filtered instead -- that
keeps rejudge's own column grafting and atomic write rather than copying them.

    setsid nohup python analysis/judge_ablation_free3.py top10 unc shuf cites pad \
        > logs/ablation_free3.log 2>&1 < /dev/null &

``--judges oss`` fills the gpt-oss seat alone; ``--judges ds_or`` fills the
DeepSeek seat alone through OpenRouter, pinned to DeepInfra, for the days the
FAU deployment is down -- that one is billed, so it needs ``--yes``:

    python analysis/judge_ablation_free3.py --judges ds_or --with-combined --yes \
        combined unc shuf cites pad

``--dry-run`` prints the plan and the cost estimate without calling anything.

``rejudge`` builds one ensemble
over every missing seat and writes only once it has all of them, so when the
DeepSeek deployment is down a free3 pass loses the gpt-oss seat along with it --
that is how `_gold_unc` and `_gold_shuf` ended up with neither on 2026-09-12,
while `_gold_cites` and `_gold_pad`, whose DeepSeek seat returned empty instead
of raising, kept theirs. Banking gpt-oss on its own first makes the DeepSeek
retry cost one seat rather than two.
"""
import sys, os, time, glob
sys.path.insert(0, 'analysis')
import oracle_experiment as oe

SKIP = '_gold_combined'
_orig = oe.targets

#: ``--judges`` values. ``oss`` is not in ``oe.PANELS``: it is a one-seat panel
#: that exists only to bank the free seat independently of the flaky one.
#: ``ds_or`` is the same seat as free3's DeepSeek, served through OpenRouter for
#: the days NHR@FAU's deployment is down. It writes a third column spelling --
#: the slug is lowercase -- which ``paper_table.SEAT_ALIASES`` resolves back
#: onto the canonical seat, so the tables read it without further work.
SEATS = {'free3': oe.PANELS['free3'], 'oss': ('openai/gpt-oss-120b',),
         'ds_or': ('deepseek/deepseek-v4-flash-0731',)}
#: OpenRouter routing for ``ds_or``. Not cosmetic: of the four providers serving
#: this slug, one is biased +13.75 marks high and one refuses 11 of 12 prompts.
#: See the fallback block in endpoints.yaml for the comparison.
PROVIDER = {'ds_or': {'order': ['DeepInfra'], 'allow_fallbacks': False,
                      'data_collection': 'deny'}}


def arm_file(tag):
    hits = glob.glob(f'{oe.D}/*_gold_{tag}.csv')
    return hits[0] if hits else None


def judge(tag, panel, provider=None, confirmed=False, skip=SKIP, dry_run=False,
          concurrency=None):
    oe.targets = lambda d=oe.D, arm='all': [
        p for p in _orig(d, arm)
        if f'_gold_{tag}.csv' == os.path.basename(p)[-len(f'_gold_{tag}.csv'):]
        and (not skip or skip not in os.path.basename(p))]
    oe.rejudge(panel=panel, arm='other', provider=provider, confirmed=confirmed,
               dry_run=dry_run, concurrency=concurrency)


args = sys.argv[1:]
which = 'free3'
if '--judges' in args:
    i = args.index('--judges')
    which = args[i + 1]
    del args[i:i + 2]
# ``_gold_combined``'s cell was published from the pre-deletion run, so re-doing
# it bought only a variance check -- until the re-generated arm turned out to
# carry no DeepSeek seat at all, and it is the baseline every other arm is
# differenced against. --with-combined judges it like any other arm.
with_combined = '--with-combined' in args
if with_combined:
    args.remove('--with-combined')
confirmed = '--yes' in args
if confirmed:
    args.remove('--yes')
dry_run = '--dry-run' in args
if dry_run:
    args.remove('--dry-run')
concurrency = None
if '--concurrency' in args:
    i = args.index('--concurrency')
    concurrency = int(args[i + 1])
    del args[i:i + 2]
skip = '' if with_combined else SKIP
panel = SEATS[which]
provider = PROVIDER.get(which)
print(f'### seats: {which} -> {", ".join(panel)}'
      + (f'   provider {provider["order"]}' if provider else ''), flush=True)

for tag in args:
    if tag == 'combined' and not with_combined:
        print(f'### skipping {tag} by request', flush=True)
        continue
    while arm_file(tag) is None:            # still generating into .part
        if dry_run:
            print(f'### _gold_{tag}: no file yet', flush=True)
            break
        print(f'### {time.strftime("%F %T")}  waiting for _gold_{tag}', flush=True)
        time.sleep(300)
    print(f'### {time.strftime("%F %T")}  {which} top-up for _gold_{tag}', flush=True)
    try:
        judge(tag, panel, provider=provider, confirmed=confirmed, skip=skip,
              dry_run=dry_run, concurrency=concurrency)
    except Exception as e:                  # a dead seat must not stop the rest
        print(f'### FAILED _gold_{tag}: {type(e).__name__}: {e}', flush=True)
print(f'### {time.strftime("%F %T")}  done', flush=True)
