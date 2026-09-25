# Inter-judge agreement

Do the three judges of `Table 5` agree with each other? One script, one table, one
figure. Everything here is an output of `judge_agreement.py`; nothing here is an input.

**Inputs live elsewhere and stay there.** The row set, the per-judge scores and the
seat-file loader all come from `../deepseek_rerun/`, which owns the re-judged panel.
This directory deliberately keeps no copy of them: a second `main_essay_table.csv` is
exactly the drift that left the draft's Tables 15 and 16 describing a panel the study
had stopped using.

| file | what it is |
|---|---|
| `judge_agreement.py` | the analysis; `--self-test`, `--no-figure`, `--boot N` |
| `judge_agreement_zb.tex` | bare `tabular`, drops into the paper |
| `judge_agreement_zb.csv` | full precision, with p-values |
| `judge_agreement_zb.pdf` / `.png` | the figure |
| `judge_agreement_zb_figure_caption.txt` | the figure's caption (the table's is inside the `.tex`) |

The `_zb` suffix is on the artefacts rather than the script: the `.tex` is copied into
`literature/paper/` and the `.pdf` is `\includegraphics`'d, so once they leave this
directory the filename is the only provenance they still carry.

## What this replaces

The paper answered this question twice already — `Table 15` in long form, `Table 16` as
its compact restatement — and both described a panel the study no longer uses: **`gpt-5-nano` in place of `gpt-oss-120b`, 26
models rather than 32, ~2,100 essays rather than 2,592**. A validation of judges that did
not produce the reported scores is worse than none, so both are replaced by the single
table this script writes.

## The row set is asserted, not assumed

The row set is not chosen here. It is read off `main_essay_table.csv` — the same parent
`merged_results_table.py` reads for Table 5 — and three assertions hold that to something
stronger than a shared filename: the model set must match exactly, the rebuilt per-judge
means must reproduce that file's `gpt_oss`/`qwen36`/`dsv4` columns, and the rebuilt
per-case medians must reproduce its `essay` column. The last is the one that bites: means
agree whenever the marginals do, but the median column only reproduces if every
`(model, index)` pair lines up.

| level | judge pair | *n* | ρ | κ*w* |
|---|---|---:|---|---|
| **model** | GPT-oss / Qwen3.6 | 32 | 0.95 | **0.92** |
| | GPT-oss / DS-V4 | 32 | 0.95 | **0.92** |
| | Qwen3.6 / DS-V4 | 32 | 0.99 | **0.82** |
| **essay** | GPT-oss / Qwen3.6 | 2,591 | 0.81 | **0.80** |
| | GPT-oss / DS-V4 | 2,591 | 0.78 | **0.79** |
| | Qwen3.6 / DS-V4 | 2,592 | 0.81 | **0.75** |

All model-level coefficients have p < 0.001. The `.tex` carries intervals at two decimals
and a caption that states the design; the `.csv` keeps full precision and the p-values.

## Reading the numbers

**Why κw is here.** ρ is invariant to a judge's overall level, and this panel's judges sit
at visibly different heights — mean score **28.4** (GPT-oss), **33.4** (Qwen3.6), **24.8**
(DS-V4). A reader who takes a judge's number at face value is affected by exactly the
offset ρ discards. The sharpest case is Qwen3.6 / DS-V4, which has the *highest*
essay-level ρ of the three pairs and the *lowest* κw: the two rank essays together and
then place them ~8.5 points apart.

**Pearson's *r* was here and was dropped.** On this data it tracked ρ closely enough to
add a column without adding a question — it answers the same "do they move together",
less robustly on a bounded 11-level scale, and it is level-invariant in the same way.
Two statistics that fail differently earn their columns; a third that fails like the
first does not.

Aggregation is what rescues the leaderboard — ρ climbs 0.80 → 0.97 once scores are
averaged within a model, the same effect `judge_reproducibility.py` measures against a
re-run rather than against another judge.

## Method

**Two things worth knowing about the statistics.** At the essay level the scores are
already ordinal (a 0–1 grid in 0.1 steps → K = 11 fixed levels), so κw is Cohen's, with
the level set pinned to all eleven rather than to the ones a pair happens to use. At the
model level the values are means, not grid points; rather than bin them to a grid built
for individual essays — lossy here, DS-V4's 32 means occupy 6 of 11 bins with 12 in one —
κw is taken in its bin-width-to-zero limit. It is the same statistic: where both forms are
defined the two agree to 0.001, which `--self-test` checks, along with the ordinal form
against `sklearn.metrics.cohen_kappa_score`.

Essay-level intervals resample **whole generation models**, not essays. 81 essays share a
generator, so an essay-wise bootstrap treats 2,592 observations as 2,592 independent draws.
Measured across the three κw intervals it comes out **3.6×, 5.7× and 5.8× too narrow** —
on Qwen3.6 / DS-V4 that is [0.727, 0.767] pretending against [0.587, 0.822] honest. Point
estimates are unaffected. For the same reason essay-level p-values are omitted rather than
computed.

## Running

```bash
conda activate gpbam
python experiments/analysis/judge_agreement_zb/judge_agreement.py
python experiments/analysis/judge_agreement_zb/judge_agreement.py --self-test
```

Writes `judge_agreement_zb.{tex,csv,pdf,png}` and the figure caption here. Both captions
are generated, not stored: each quotes coefficients, and a caption that can drift from the
artefact it sits under is worse than none. `--no-figure` skips the plot, `--boot N`
trades interval precision for time (default 10,000 resamples; a full run is ~50 s).

The figure is the one thing the table cannot show: **where** the disagreement lives. Its
top row is the essay-level joint distribution on the judges' own 11-point grid — the very
matrix κw is computed from, so the off-diagonal mass the statistic charges for is visible,
and visibly one-sided. The bottom row is the same pair at the model level against the
identity line, where that one-sidedness has become a clean vertical offset.
