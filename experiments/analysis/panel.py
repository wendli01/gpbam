"""The paper's model panel, in one place.

Every table and figure over "the models" means the same 32 rows, and that set is
already written down: ``main_results_table.csv``, the artefact the main results
table is rendered from. Scripts used to restate it by hardcoding the two
``RETIRED`` exclusions, which is how the contamination probe quietly ran on 31
models and the within-model test on 29.

Read the roster from the table instead. A model that joins the table and has no
data wired into a probe then shows up as a named gap rather than a smaller *n*.
"""

import os

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
MAIN_TABLE = os.path.join(_HERE, 'deepseek_rerun', 'results', 'main_results_table.csv')


def key(model):
    """Join key across runs: last path segment, lower-case, quantisation dropped.

    Runs disagree on how they spell a model. The recitation sweep used the API
    alias ``qwen/qwen3.6-35b-a3b`` where the essay run used the self-hosted
    ``Qwen/Qwen3.6-35B-A3B-FP8``; joining on the raw string drops that model.
    Same normalisation as ``plot_recitation_vs_essay._key``.
    """
    s = str(model).split('/')[-1].lower().replace('_', '-')
    for suffix in ('-fp8-dynamic', '-fp8-block', '-fp8'):
        s = s.replace(suffix, '')
    return s.replace('-a3b', '')


def roster(path=MAIN_TABLE):
    """The main table's models: one row per model, with its join key."""
    m = pd.read_csv(path, usecols=['model', 'short'])
    m['key'] = m.model.map(key)
    return m


def check(keys, what, path=MAIN_TABLE):
    """Name the roster models *keys* misses, and anything in *keys* that is not
    on the roster. Returns the roster, so a caller can filter with it."""
    m = roster(path)
    keys = set(keys)
    missing = sorted(set(m.key) - keys)
    extra = sorted(keys - set(m.key))
    if missing:
        print(f'{what}: no data for {len(missing)} of the {len(m)} models in the '
              f'main table: {", ".join(missing)}')
    if extra:
        print(f'{what}: not in the main table, dropped: {", ".join(extra)}')
    if not missing and not extra:
        print(f'{what}: all {len(m)} models in the main table, and no others')
    return m
