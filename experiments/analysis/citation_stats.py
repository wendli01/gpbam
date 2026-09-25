"""Descriptive statistics for the statutory references in GPBam.

Answers two questions the draft currently asserts rather than measures:

1. What do the GPBam cases actually cite -- in the facts and in the reference
   solutions -- by law book, citation form (``§`` vs ``Art.``) and jurisdiction?
2. How much of that does the current metric see?  ``src.scoring`` uses refex,
   which resolves no ``Art.``-form citation, so everything constitutional and
   everything Bavarian is invisible to ``legal_ref_sim``.

Run from ``experiments/``::

    python analysis/citation_stats.py

Writes tables to ``analysis/out/*.csv`` and figures to ``analysis/out/*.pdf``.
"""

import json
import os
import warnings
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import refs as refs_mod

OUT = 'analysis/out'
GPBAM = 'data/gpbam_w_rubric.json'
CORPUS_INDEX = 'analysis/out/corpus_norms.csv'


# --------------------------------------------------------------------------
# data


def load_gpbam(path=GPBAM):
    d = json.load(open(path))
    order = sorted(d['facts'], key=lambda x: int(x))
    return ([d['facts'][k] for k in order],
            [d['solutions'][k] for k in order])


def load_corpus_books(path=CORPUS_INDEX):
    """Law books present in the federal gesetze-im-internet dump."""
    if not os.path.exists(path):
        warnings.warn(f'{path} missing -- run analysis/build_corpus_index.py first')
        return None, None
    c = pd.read_csv(path, dtype=str).fillna('')
    c['key'] = c.law_book.map(refs_mod.corpus_key)
    books = set(c.key)
    norms = {(r.key, r.section.lower()) for r in c.itertuples() if r.section}
    return books, norms


def cite_frame(texts, part):
    """One row per citation occurrence in *texts*.

    Two keys per row on purpose. ``book_key`` is the surface abbreviation and is
    what the by-book figure groups and labels on. ``norm_book`` runs it through
    :func:`refs.corpus_key`, which folds spelling variants onto one corpus
    ``jurabk`` -- the same normalisation :mod:`reference_coverage` applies, so
    the per-case norm counts here and the counts behind
    ``tab:reference_coverage`` are one extraction reported twice rather than two
    that happen to be close.
    """
    rows = []
    for case_id, text in enumerate(texts):
        for c in refs_mod.canonicalise(refs_mod.extract(text)):
            rows.append(dict(case=case_id, part=part, book=c.book,
                             book_key=c.book.lower(), section=c.section, form=c.form,
                             norm_book=refs_mod.corpus_key(c.book),
                             norm_section=c.section.lower()))
    return pd.DataFrame(rows)


def refex_frame(texts, part):
    """The same texts as seen by the extractor the scoring pipeline uses."""
    from refex.extractor import RefExtractor
    from refex.errors import RefExError
    ex = RefExtractor()
    rows = []
    for case_id, text in enumerate(texts):
        try:
            _, markers = ex.extract(text)
        except RefExError:
            continue
        for mk in markers:
            for r in mk.get_references():
                if r.book and r.section:
                    rows.append(dict(case=case_id, part=part,
                                     book_key=r.book.lower(), section=str(r.section).lower()))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# tables


def per_case_table(df_facts, df_sol):
    """Citations, distinct norms and distinct books per case."""
    def agg(df, label):
        g = df.groupby('case').agg(**{
            f'{label}_citations': ('book', 'size'),
            f'{label}_books': ('norm_book', 'nunique'),
        })
        g[f'{label}_norms'] = df.groupby('case').apply(
            lambda d: len(set(zip(d.norm_book, d.norm_section))), include_groups=False)
        return g
    t = agg(df_sol, 'solution').join(agg(df_facts, 'facts'), how='outer').fillna(0)
    return t.astype(int)


def book_table(df, corpus_books, top=25):
    g = df.groupby('book_key').agg(citations=('book', 'size'),
                                   display=('book', lambda s: s.mode().iloc[0]),
                                   cases=('case', 'nunique'))
    g['art_share'] = df.assign(is_art=df.form.eq('Art')).groupby('book_key').is_art.mean()
    g['jurisdiction'] = [refs_mod.jurisdiction(b, corpus_books) for b in g.index]
    g['in_federal_corpus'] = [refs_mod.corpus_key(b) in corpus_books if corpus_books else None
                              for b in g.index]
    return g.sort_values('citations', ascending=False).head(top)


def coverage_table(df_ours, df_refex):
    """What share of the reference solutions' citations does refex resolve?"""
    ours = set(zip(df_ours.case, df_ours.book_key, df_ours.section))
    theirs = set(zip(df_refex.case, df_refex.book_key, df_refex.section))
    by_form = df_ours.groupby('form').apply(
        lambda d: len(set(zip(d.case, d.book_key, d.section))), include_groups=False)
    rows = [
        dict(metric='distinct case-level norms, this extractor', value=len(ours)),
        dict(metric='distinct case-level norms, refex (scoring pipeline)', value=len(theirs)),
        dict(metric='  of which also found here', value=len(ours & theirs)),
        dict(metric='Art.-form norms (invisible to refex)', value=int(by_form.get('Art', 0))),
        dict(metric='§-form norms', value=int(by_form.get('Par', 0))),
    ]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# figures

PALETTE = {'federal': '#3E6B8A', 'state': '#B3762F', 'unknown': '#9AA3A0'}


def fig_books(bt, path):
    # smaller canvas at unchanged font sizes -> larger type once the panel is
    # scaled to the column width; slightly flatter than square
    fig, ax = plt.subplots(figsize=(6.0, 5.1), dpi=200)
    bt = bt.iloc[::-1]
    y = np.arange(len(bt))
    art = bt.citations * bt.art_share
    par = bt.citations - art
    colors = [PALETTE[j] for j in bt.jurisdiction]
    ax.barh(y, par, color=colors, edgecolor='white', linewidth=.6, label='§ form')
    ax.barh(y, art, left=par, color=colors, alpha=.45, edgecolor='white',
            linewidth=.6, hatch='///', label='Art. form')
    ax.set_yticks(y)
    ax.set_yticklabels(bt.display, fontsize=9)
    ax.set_xlabel('Citations in the 81 reference solutions')
    ax.set_title('What GPBam cases cite', loc='left', fontweight='bold')
    handles = [plt.Rectangle((0, 0), 1, 1, color=PALETTE['federal']),
               plt.Rectangle((0, 0), 1, 1, color=PALETTE['state']),
               plt.Rectangle((0, 0), 1, 1, color='#666666', alpha=.45, hatch='///')]
    ax.legend(handles, ['federal law', 'state law', 'cited as “Art. …” (refex-invisible)'],
              frameon=False, fontsize=8, loc='lower right')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='x', alpha=.25, linewidth=.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    fig.savefig(path.replace('.pdf', '.png'), bbox_inches='tight')
    plt.close(fig)


def fig_per_case(t, path):
    """Both distributions on one axis.

    The two ranges are disjoint -- no case names more than 14 norms, no
    reference solution fewer than 20 -- so a shared x-axis separates them
    without any overlap and makes the gap the point of the panel.
    """
    fig, ax = plt.subplots(figsize=(5.4, 4.0), dpi=200)
    hi = int(t[['solution_norms', 'facts_norms']].max().max())
    bins = np.arange(0, hi + 5, 4)
    peaks = {}
    for col, label, color in (
            ('facts_norms', 'case facts (“Sachverhalt”)', PALETTE['state']),
            ('solution_norms', 'reference solution', PALETTE['federal'])):
        counts, _, _ = ax.hist(t[col], bins=bins, color=color, alpha=.85,
                               edgecolor='white', linewidth=.7, label=label,
                               zorder=3)
        peaks[col] = counts.max()
    # each median label sits just above its own distribution, not at the top of
    # the axes -- the two peaks differ by a factor of five
    for col in ('facts_norms', 'solution_norms'):
        med = t[col].median()
        ax.axvline(med, color='#22282B', linestyle='--', linewidth=1, zorder=4)
        top = ax.get_ylim()[1]
        ax.annotate(f'median {med:.0f}', (med, min(peaks[col] + top * .04, top * .86)),
                    xytext=(4, 0), textcoords='offset points', fontsize=8.5,
                    va='bottom', color='#22282B')
    ax.set_xlabel('distinct (law book, section) pairs cited')
    ax.set_ylabel('cases')
    ax.set_title('What a case names vs. what its solution needs',
                 loc='left', fontweight='bold', fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, loc='upper center')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=.25, linewidth=.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    fig.savefig(path.replace('.pdf', '.png'), bbox_inches='tight')
    plt.close(fig)


# --------------------------------------------------------------------------


def main():
    os.makedirs(OUT, exist_ok=True)
    facts, sols = load_gpbam()
    corpus_books, corpus_norms = load_corpus_books()

    df_sol = cite_frame(sols, 'solution')
    df_facts = cite_frame(facts, 'facts')
    df_refex = refex_frame(sols, 'solution')

    print(f'reference solutions: {len(df_sol)} citation occurrences, '
          f'{df_sol.book_key.nunique()} distinct law books')
    print(f'case facts:          {len(df_facts)} citation occurrences, '
          f'{df_facts.book_key.nunique()} distinct law books')

    per_case = per_case_table(df_facts, df_sol)
    per_case.to_csv(f'{OUT}/citations_per_case.csv')
    print('\nper-case distribution')
    print(per_case.describe().loc[['mean', '50%', 'min', 'max']].round(1).to_string())

    bt = book_table(df_sol, corpus_books)
    bt.to_csv(f'{OUT}/top_books_solutions.csv')
    print('\ntop law books in the reference solutions')
    print(bt[['display', 'citations', 'cases', 'art_share', 'jurisdiction',
              'in_federal_corpus']].round(2).to_string())

    bf = book_table(df_facts, corpus_books, top=15)
    bf.to_csv(f'{OUT}/top_books_facts.csv')
    print('\ntop law books in the case facts')
    print(bf[['display', 'citations', 'cases', 'art_share', 'jurisdiction']].round(2).to_string())

    cov = coverage_table(df_sol, df_refex)
    cov.to_csv(f'{OUT}/refex_coverage.csv', index=False)
    print('\nwhat the scoring pipeline sees')
    print(cov.to_string(index=False))

    # jurisdiction split of the citation mass
    df_sol['jurisdiction'] = [refs_mod.jurisdiction(b, corpus_books) for b in df_sol.book_key]
    js = df_sol.jurisdiction.value_counts(normalize=True).round(3)
    print('\njurisdiction split of citation occurrences in the solutions:')
    print(js.to_string())

    # corpus reachability ceiling
    if corpus_norms:
        gold = {(refs_mod.corpus_key(b), s) for b, s in zip(df_sol.book_key, df_sol.section)}
        in_corpus = {n for n in gold if n in corpus_norms}
        print(f'\ndistinct gold norms: {len(gold)}; present in the federal corpus: '
              f'{len(in_corpus)} ({100 * len(in_corpus) / len(gold):.1f}%)')
        pd.DataFrame([dict(distinct_gold_norms=len(gold), in_federal_corpus=len(in_corpus))]
                     ).to_csv(f'{OUT}/corpus_reachability.csv', index=False)

    fig_books(bt, f'{OUT}/gpbam_citations_by_book.pdf')
    fig_per_case(per_case, f'{OUT}/gpbam_citations_per_case.pdf')
    print(f'\nfigures written to {OUT}/')


if __name__ == '__main__':
    main()
