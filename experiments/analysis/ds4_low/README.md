# DeepSeek at `low`, judged at `low` (2026-09-13)

The scripts that produced the DeepSeek low-effort judge seat for
`oracle_rag/DS4-ablation-low/` and `rag_titled_combined_ls_text/DS4-ablation-low/`.
Findings and numbers: `DS4-ablation-rerun.md` (repo root), "DeepSeek at `low`".

```bash
cd experiments
python analysis/ds4_low/judge_low.py <arm.csv> <side.csv>              # DeepSeek seat at low on DeepInfra -> side file
python analysis/ds4_low/judge_low_fill.py <arm.csv> <side.csv> [...]   # retry rows whose score did not parse
python analysis/ds4_low/swap_judge_columns.py <arm.csv> <side.csv> [...]  # low seat canonical, high kept as "..., effort high)"
python analysis/ds4_low/judge_low_gwdg.py                                 # same seat on GWDG (validation, combined arm)
python analysis/ds4_low/cells.py                                        # every cell, DeepSeek judge high vs low
```

`judge_low.py` needs both arguments; it was run once per arm. The swap is
idempotent per file (it refuses to swap twice). The ls_text arm had no
high-effort DeepSeek seat (OpenRouter ran out of credit, 402), so its low seat
was grafted directly.
