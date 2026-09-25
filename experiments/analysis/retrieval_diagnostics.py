"""Does the RAG condition retrieve the statutes the reference solution cites?

Everything here is recovered from the stored with-RAG result CSV: ``evaluate_model``
saves the full generation prompt as ``qa_prompt``, and ``qa.AnswerGenerator.format_context``
writes each retrieved passage as ``[SOURCE i: {law_book} {paragraph} ({title})]``.
So the retrieved set for every essay is recoverable without re-running retrieval.

Each gold norm falls into exactly one of three buckets, which separates the three
hypotheses the single RAG condition otherwise conflates:

  not-in-corpus   the norm is absent from the federal gesetze-im-internet dump,
                  so no retriever over this corpus could ever return it
  missed          present in the corpus, not in this essay's top-5
  retrieved       handed to the model in its own prompt

Two gold sets are available::

    python analysis/retrieval_diagnostics.py                  # full extractor (analysis/refs.py)
    python analysis/retrieval_diagnostics.py --gold refex     # refex only

The ``refex`` mode restricts the gold set to the citations the scoring pipeline
itself resolves, so the failure-mode numbers are consistent with the published
``legal_ref_sim`` column.  It is the version to quote in the paper: refex's
blindness to the ``Art.`` form (and therefore to GG and to Bavarian state law)
then applies equally to the metric and to this diagnostic, and becomes a single
stated caveat instead of a discrepancy between two numbers.
"""

import argparse
import json
import os
import re
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import refs as refs_mod

OUT = 'analysis/out'
#: The published with-RAG run this script was written against has been removed:
#: it was too broken to recover anything from. Every number below -- recall@5,
#: precision of the five retrieved passages, the reachability figures -- is a
#: diagnostic of *that* retrieval, five passages from rewritten queries over the
#: federal-only corpus, and none of it describes the system the paper now ships.
#:
#: The live equivalents are recall_battery.py (retrieval quality per pipeline)
#: and corpus_rag_run.py (corpus contrast). This is kept rather than deleted
#: because the reachability analysis is worth re-pointing at a current arm, but
#: that is not a repoint of one constant: the current arms build their context
#: without the `[SOURCE n: ...]` markers parse_sources() keys on, so it needs a
#: reader for the new format first.
RAG_CSV = None
CORPUS_INDEX = 'analysis/out/corpus_norms.csv'
GPBAM = 'data/gpbam_w_rubric.json'

SOURCE_RE = re.compile(r'\[SOURCE \d+: (.*?) \(')
BOILERPLATE_RE = re.compile(
    r'Eingangsformel|Inhalts(?:verzeichnis|übersicht)|Schlussformel|Anlage|Anhang|Präambel',
    re.IGNORECASE)
SECTION_RE = re.compile(r'(?:§+|Art\.?)\s*(\d+\s?[a-z]?)\s*$')


def parse_sources(prompt):
    """``[SOURCE 1: BImSchG § 14 (title)]`` -> ``('bimschg', '14')``.

    The header is *not* two whitespace-separated fields: real law books carry
    spaces and version years (``MalerLackAusbV 2021 § 22``) and the paragraph
    slot often holds structural labels instead of a section (``FeuAO
    Eingangsformel``).  Split on the trailing section marker instead.
    """
    out = []
    if not isinstance(prompt, str):
        return out
    for header in SOURCE_RE.findall(prompt):
        m = SECTION_RE.search(header)
        if m:
            book = header[:m.start()].strip()
            section = re.sub(r'\s+', '', m.group(1))
        else:
            book, section = re.split(r'\s+(?=\S+$)', header, maxsplit=1) if ' ' in header else (header, '')
            section = ''
        out.append(dict(raw=header, book=refs_mod.corpus_key(book), section=section.lower(),
                        boilerplate=bool(BOILERPLATE_RE.search(header))))
    return out


def normalise_typography(text):
    """Fold the Unicode spaces and dashes some models emit inside citations.

    gpt-oss-120b writes ``§ 31 BauGB`` with U+202F NARROW NO-BREAK SPACE.
    refex resolves none of those, so its citations are dropped from
    ``legal_ref_sim`` for a purely typographic reason -- 15 references in a single
    essay in the case checked.  Normalising costs nothing and is applied here to
    every model equally.
    """
    if not isinstance(text, str):
        return text
    for ch in '    ⁠':
        text = text.replace(ch, ' ')
    return text.replace('‑', '-').replace('‐', '-')


def refex_norms(text, normalise=True):
    """The citations the scoring pipeline resolves, as ``(book, section)`` pairs."""
    from refex.extractor import RefExtractor
    from refex.errors import RefExError
    ex = RefExtractor()
    if normalise:
        text = normalise_typography(text)
    try:
        _, markers = ex.extract(text)
    except RefExError:
        return set()
    return {(refs_mod.corpus_key(r.book), str(r.section).lower())
            for mk in markers for r in mk.get_references() if r.book and r.section}


def main(gold='full'):
    os.makedirs(OUT, exist_ok=True)
    suffix = '' if gold == 'full' else f'_{gold}'
    print(f'gold citation set: {gold}\n')

    d = json.load(open(GPBAM))
    sols = [d['solutions'][k] for k in sorted(d['solutions'], key=lambda x: int(x))]
    gold_norms, gold_books = {}, {}
    for i, s in enumerate(sols):
        if gold == 'refex':
            gold_norms[i] = refex_norms(s)
        else:
            cits = refs_mod.canonicalise(refs_mod.extract(s))
            gold_norms[i] = {(refs_mod.corpus_key(c.book), c.section.lower()) for c in cits}
        gold_books[i] = {b for b, _ in gold_norms[i]}

    corpus = pd.read_csv(CORPUS_INDEX, dtype=str).fillna('')
    corpus['key'] = corpus.law_book.map(refs_mod.corpus_key)
    corpus_norms = {(r.key, r.section.lower()) for r in corpus.itertuples() if r.section}
    corpus_books = set(corpus.key)
    print(f'corpus: {len(corpus)} norms, {len(corpus_books)} law books, '
          f'{corpus.is_boilerplate.astype(int).sum()} structural (Eingangsformel/Anlage/...)')

    if RAG_CSV is None:
        raise SystemExit(
            'retrieval_diagnostics reads the removed with-RAG run; see the note '
            'on RAG_CSV. Use recall_battery.py or corpus_rag_run.py instead.')
    df = pd.read_csv(RAG_CSV)
    df['sources'] = df.qa_prompt.map(parse_sources)
    print(f'{len(df)} with-RAG essays, {df.sources.map(len).sum()} retrieved passages')

    flat = [s for row in df.sources for s in row]
    print(f'structural/boilerplate share of retrieved passages: '
          f'{100 * np.mean([s["boilerplate"] for s in flat]):.1f}%')

    # ---- per-essay retrieval quality
    rows = []
    for r in df.itertuples():
        i = int(r.index)
        ret_norms = {(s['book'], s['section']) for s in r.sources if s['section']}
        ret_books = {s['book'] for s in r.sources}
        g_n, g_b = gold_norms[i], gold_books[i]
        rows.append(dict(
            model=r.model, case=i,
            gold_norms=len(g_n), gold_books=len(g_b),
            hit_norms=len(ret_norms & g_n), hit_books=len(ret_books & g_b),
            retrieved_norms=len(ret_norms),
            boilerplate=sum(s['boilerplate'] for s in r.sources),
        ))
    per_essay = pd.DataFrame(rows)
    per_essay.to_csv(f'{OUT}/retrieval_per_essay{suffix}.csv', index=False)

    print('\n--- retrieval sanity check, over all %d essays ---' % len(per_essay))
    print(f'essays with >=1 retrieved norm the solution also cites: '
          f'{100 * (per_essay.hit_norms > 0).mean():.1f}%')
    print(f'essays with >=1 retrieved law book the solution also cites: '
          f'{100 * (per_essay.hit_books > 0).mean():.1f}%')
    print(f'mean gold norms recalled per essay: {per_essay.hit_norms.mean():.2f} '
          f'(of {per_essay.gold_norms.mean():.1f} cited, from 5 passages)')
    recall = per_essay.hit_norms.sum() / per_essay.gold_norms.sum()
    precision = per_essay.hit_norms.sum() / (5 * len(per_essay))
    ceiling = 5 / per_essay.gold_norms.mean()
    # a retriever drawing 5 passages uniformly from the corpus would hit a gold
    # norm with probability (reachable gold norms for this case) / (corpus size)
    reachable_share = np.mean([len(g & corpus_norms) / len(g) for g in gold_norms.values() if g])
    random_precision = per_essay.gold_norms.mean() * reachable_share / len(corpus_norms)
    print(f'recall@5 over gold norms: {100 * recall:.2f}%   '
          f'(structural ceiling with 5 slots and {per_essay.gold_norms.mean():.0f} '
          f'gold norms per case: {100 * ceiling:.1f}%)')
    print(f'precision of the 5 retrieved passages: {100 * precision:.2f}%   '
          f'(uniform-random draw from the corpus would give {100 * random_precision:.3f}%)')

    # ---- three-way decomposition over distinct gold norms per case
    dec = []
    retrieved_by_case = {}
    for i in gold_norms:
        sub = df[df['index'] == i]
        retrieved_by_case[i] = {(s['book'], s['section'])
                                for row in sub.sources for s in row if s['section']}
    for i, g in gold_norms.items():
        for n in g:
            if n not in corpus_norms:
                bucket = 'not in corpus'
            elif n in retrieved_by_case[i]:
                bucket = 'retrieved'
            else:
                bucket = 'in corpus, missed'
            dec.append(dict(case=i, book=n[0], section=n[1], bucket=bucket,
                            jurisdiction=refs_mod.jurisdiction(n[0], corpus_books)))
    dec = pd.DataFrame(dec)
    dec.to_csv(f'{OUT}/gold_norm_buckets{suffix}.csv', index=False)

    print('\n--- where do the reference solutions\' norms end up? ---')
    print('(union over every retrieval this case received, i.e. an upper bound '
          'on what any single essay saw)')
    share = dec.bucket.value_counts(normalize=True).mul(100).round(1)
    cnt = dec.bucket.value_counts()
    for b in ['not in corpus', 'in corpus, missed', 'retrieved']:
        print(f'   {b:<20} {cnt.get(b, 0):>5}  ({share.get(b, 0):>4.1f}%)')

    reachable_only = dec[dec.bucket != 'not in corpus']
    print(f'\nretriever performance on the reachable half alone: '
          f'{100 * (reachable_only.bucket == "retrieved").mean():.1f}% of the gold norms that '
          f'ARE in the corpus were ever retrieved for their case')

    print('\nby jurisdiction:')
    print(pd.crosstab(dec.jurisdiction, dec.bucket, normalize='index').mul(100).round(1).to_string())

    # ---- figure
    per_case_reach = dec.groupby('case').apply(
        lambda d: (d.bucket != 'not in corpus').mean(), include_groups=False)
    print(f'\nper-case reachability: median {100 * per_case_reach.median():.0f}%, '
          f'{int((per_case_reach < .25).sum())} of 81 cases have under a quarter of their '
          f'gold norms in the corpus, {int((per_case_reach > .75).sum())} have over three quarters')

    fig_buckets(dec, f'{OUT}/retrieval_buckets{suffix}.pdf')
    fig_case_reachability(dec, f'{OUT}/retrieval_case_reachability{suffix}.pdf')
    print(f'\nfigures written to {OUT}/')


COLORS = {'retrieved': '#2F6B57', 'in corpus, missed': '#C08A2E', 'not in corpus': '#9E4A44'}
ORDER = ['retrieved', 'in corpus, missed', 'not in corpus']


def fig_buckets(dec, path):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), dpi=200,
                             gridspec_kw={'width_ratios': [1, 1.5]})

    # overall
    ax = axes[0]
    share = dec.bucket.value_counts(normalize=True).mul(100)
    left = 0
    for b in ORDER:
        v = share.get(b, 0)
        ax.barh([0], [v], left=left, height=.45, color=COLORS[b],
                edgecolor='white', linewidth=.8)
        if v > 8:
            ax.text(left + v / 2, 0, f'{v:.0f}%', ha='center', va='center',
                    color='white', fontsize=10, fontweight='bold')
        else:
            ax.annotate(f'{v:.1f}%', (left + v / 2, .26), ha='center', va='bottom',
                        fontsize=9, color=COLORS[b], fontweight='bold')
        left += v
    ax.set_yticks([])
    ax.set_ylim(-.6, .6)
    ax.set_xlim(0, 100)
    ax.set_xlabel('% of distinct gold norms')
    ax.set_title('All 81 cases', fontsize=10, loc='left')
    ax.spines[['top', 'right', 'left']].set_visible(False)

    # by jurisdiction
    ax = axes[1]
    ct = pd.crosstab(dec.jurisdiction, dec.bucket, normalize='index').mul(100)
    ct = ct.reindex(columns=[b for b in ORDER if b in ct.columns], fill_value=0)
    ct = ct.reindex([j for j in ['federal', 'state', 'unknown'] if j in ct.index])
    y = np.arange(len(ct))
    left = np.zeros(len(ct))
    for b in ct.columns:
        ax.barh(y, ct[b], left=left, color=COLORS[b], edgecolor='white', linewidth=.8, label=b)
        for yi, (v, l) in enumerate(zip(ct[b], left)):
            if v > 6:
                ax.text(l + v / 2, yi, f'{v:.0f}%', ha='center', va='center',
                        color='white', fontsize=9, fontweight='bold')
        left += ct[b].values
    ax.set_yticks(y)
    ax.set_yticklabels([f'{j}\n(n={int((dec.jurisdiction == j).sum())})' for j in ct.index], fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_xlabel('% of distinct gold norms')
    ax.set_title('By jurisdiction of the cited norm', fontsize=10, loc='left')
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc='upper center',
              bbox_to_anchor=(.5, -.28))
    fig.suptitle('Can the retriever reach the norms the reference solution cites?',
                 x=.01, ha='left', fontweight='bold', fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, .93])
    fig.savefig(path, bbox_inches='tight')
    fig.savefig(path.replace('.pdf', '.png'), bbox_inches='tight')
    plt.close(fig)


def fig_case_reachability(dec, path):
    """Per case: what share of its gold norms could retrieval reach at all?

    The spread matters more than the mean.  A federal-heavy Baurecht case has
    most of its norms in the corpus; a Polizeirecht case running on PAG and
    LStVG has almost none, so no amount of retrieval tuning can help it.
    """
    g = dec.groupby('case').apply(
        lambda d: pd.Series({
            'reachable': (d.bucket != 'not in corpus').mean() * 100,
            'state_share': (d.jurisdiction == 'state').mean() * 100,
            'n': len(d)}), include_groups=False).sort_values('reachable')
    fig, ax = plt.subplots(figsize=(8.4, 3.6), dpi=200)
    x = np.arange(len(g))
    ax.bar(x, g.reachable, color='#C08A2E', width=.9, edgecolor='none', label='in the federal corpus')
    ax.bar(x, 100 - g.reachable, bottom=g.reachable, color='#9E4A44', width=.9,
           edgecolor='none', label='outside it (state and EU law)')
    ax.axhline(g.reachable.median(), color='#22282B', linestyle='--', linewidth=1)
    ax.annotate(f'median {g.reachable.median():.0f}%', (len(g) * .02, g.reachable.median() + 2),
                fontsize=8.5)
    ax.set_xlim(-.5, len(g) - .5)
    ax.set_ylim(0, 100)
    ax.set_xlabel('the 81 GPBam cases, ordered by reachability')
    ax.set_ylabel('% of gold norms')
    ax.set_title('About half of every case lies outside the retrieval corpus',
                 loc='left', fontsize=11, fontweight='bold')
    ax.legend(frameon=False, fontsize=8.5, loc='lower right')
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    fig.savefig(path.replace('.pdf', '.png'), bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gold', choices=['full', 'refex'], default='full',
                    help="gold citation set: 'full' uses analysis/refs.py, 'refex' restricts "
                         "to what the scoring pipeline resolves (use this for the paper)")
    main(**vars(ap.parse_args()))
