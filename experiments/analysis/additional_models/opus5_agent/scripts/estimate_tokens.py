"""Estimate what the Opus 5 essays would have cost through the API.

The agent harness reports no usage record, so this row has no measured tokens and
no bill. Every other row in the price/performance figure carries one, so leaving
Opus off that axis would silently drop the most expensive model in the study.
What follows is an estimate, and it is built to be checkable rather than tight.

Prompt tokens: not estimated at all
-----------------------------------
`claude-haiku-4.5` answered the *same* 81 prompts through the API, and Anthropic
tokenises with one tokeniser across the family, so its recorded `prompt_tokens`
per case is Opus's prompt_tokens per case. This is a lookup, not a model.

Completion tokens: visible text measured, thinking carried over
---------------------------------------------------------------
Two parts, because they scale with different things.

* *Visible* output is measured from the essay itself. Claude's chars-per-token
  ratio on this material is read off the same haiku run -- 8,958 prompt chars
  against 3,220 prompt tokens, so 2.782 chars/token on German legal prose. The
  essays and the prompts are the same register, so the ratio transfers.
* *Thinking* tokens are billed as output and are invisible here. They are taken
  from haiku on the same case index. The assumption is that the thinking budget
  tracks the difficulty of the case rather than the length of the answer -- both
  models ran at `reasoning_effort='high'` on identical prompts. On haiku that
  share is 53% of output, mean 10,479 tokens.

The alternative assumption -- thinking scales with output length, i.e. Opus
thinks 2.3x as much because it writes 2.3x as much -- is reported alongside as
`completion_tokens_proportional`, and prices roughly 45% higher. Both are in the
CSV so the figure's sensitivity to the choice is one subtraction away.

Price: OpenRouter list for `anthropic/claude-opus-5`, $5.00 / $25.00 per 1M
tokens, fetched 2026-08-21. Exactly 5x `claude-haiku-4.5`, in both directions.

    python estimate_tokens.py
"""
import json
from pathlib import Path

import pandas as pd

D = Path(__file__).resolve().parent.parent
ROOT = Path(__file__).resolve().parents[5]
PUB = ROOT / 'experiments/zubaers_result/essay_writing/without_rag/ji2/no_rag_ji2_result.csv'
CALIBRATOR = 'anthropic/claude-haiku-4.5'
PRICE_IN, PRICE_OUT = 5.00, 25.00                      # USD per 1M, OpenRouter list

pub = pd.read_csv(PUB, usecols=['index', 'model', 'answer', 'qa_prompt',
                                'prompt_tokens', 'completion_tokens'])
cal = pub[pub.model == CALIBRATOR].drop_duplicates('index').set_index('index').sort_index()
chars_per_token = cal.qa_prompt.str.len().sum() / cal.prompt_tokens.sum()
cal_visible = cal.answer.fillna('').str.len() / chars_per_token
cal_thinking = cal.completion_tokens - cal_visible
assert (cal_thinking > 0).all(), 'calibrator has cases whose visible text exceeds its output'

ess = pd.DataFrame([json.loads(l) for l in open(D / 'opus5_agent.jsonl')]).set_index('index')
ess = ess.sort_index()

out = pd.DataFrame(index=ess.index)
out['chars'] = ess.chars
out['prompt_tokens'] = cal.prompt_tokens.reindex(ess.index)          # measured, not estimated
out['visible_tokens'] = (ess.chars / chars_per_token).round()
out['thinking_tokens'] = cal_thinking.reindex(ess.index).round()
out['completion_tokens'] = out.visible_tokens + out.thinking_tokens
out['completion_tokens_proportional'] = (
    out.visible_tokens * (cal.completion_tokens.sum() / cal_visible.sum())).round()
out['cost_usd'] = (out.prompt_tokens * PRICE_IN + out.completion_tokens * PRICE_OUT) / 1e6
out['cost_usd_proportional'] = (
    out.prompt_tokens * PRICE_IN + out.completion_tokens_proportional * PRICE_OUT) / 1e6

out.to_csv(D / 'token_estimate.csv')
n81 = 81
print(f'calibrator {CALIBRATOR}: {chars_per_token:.4f} chars/token, '
      f'thinking {cal_thinking.mean():.0f}/essay ({cal_thinking.sum() / cal.completion_tokens.sum():.0%} of output)')
print(f'{len(out)} essays, mean {out.chars.mean():.0f} chars '
      f'({out.chars.mean() / cal.answer.str.len().mean():.2f}x the calibrator)')
print(out[['prompt_tokens', 'visible_tokens', 'thinking_tokens',
           'completion_tokens', 'cost_usd']].mean().round(3).to_string())
print(f'\nmeasured over {len(out)} essays: ${out.cost_usd.sum():.2f}')
print(f'extrapolated to {n81}:      ${out.cost_usd.mean() * n81:.2f}  '
      f'(proportional-thinking variant: ${out.cost_usd_proportional.mean() * n81:.2f})')
print(f'wrote {D / "token_estimate.csv"}')
