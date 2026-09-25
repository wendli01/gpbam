"""Does reasoning_effort (esp. 'max') apply to DeepSeek-V4-Flash on NHR@FAU?"""
import os, time, openai
from dotenv import load_dotenv
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
load_dotenv(ROOT / '.env')
c = openai.OpenAI(api_key=os.environ['API_KEY_FAU'],
                  base_url='https://hub.nhr.fau.de/api/llmgw/v1', timeout=1800)
Q = [{'role': 'user', 'content':
      'Ein Bauherr errichtet im Aussenbereich einen 30 m hohen Mobilfunkmast. '
      'Pruefe die planungsrechtliche Zulaessigkeit nach § 35 BauGB im Gutachtenstil '
      'und nenne das Ergebnis.'}]
MODEL = 'deepseek-ai/DeepSeek-V4-Flash'
print(f"model: {MODEL}\n")
print(f"{'effort':<12}{'compl':>8}{'reason_tok':>11}{'reason_chars':>13}{'answer_chars':>13}{'sec':>7}")
res = {}
for eff in ['low', 'medium', 'high', 'xhigh', 'max', None]:
    kw = {} if eff is None else {'reasoning_effort': eff}
    t0 = time.time()
    try:
        r = c.chat.completions.create(model=MODEL, messages=Q, seed=1, temperature=0, **kw)
    except Exception as e:
        print(f"{str(eff):<12} REJECTED  {type(e).__name__}: {str(e)[:150]}")
        continue
    m = r.choices[0].message; md = m.model_dump()
    d = getattr(r.usage, 'completion_tokens_details', None)
    rt = getattr(d, 'reasoning_tokens', None) if d else None
    rc = md.get('reasoning_content') or md.get('reasoning') or ''
    res[str(eff)] = (r.usage.completion_tokens, len(rc), m.content or '')
    print(f"{str(eff):<12}{r.usage.completion_tokens:>8}{str(rt):>11}{len(rc):>13}"
          f"{len(m.content or ''):>13}{time.time()-t0:>7.1f}")
print("\nidentical answers between efforts? (identical => effort ignored)")
ks = list(res)
for i in range(len(ks)):
    for j in range(i + 1, len(ks)):
        print(f"  {ks[i]:>7} vs {ks[j]:<7} same_answer={res[ks[i]][2]==res[ks[j]][2]}  "
              f"reason_chars {res[ks[i]][1]} vs {res[ks[j]][1]}")
