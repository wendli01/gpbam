import os, openai
from dotenv import load_dotenv
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
load_dotenv(ROOT / '.env')
c=openai.OpenAI(api_key=os.environ['API_KEY_FAU'], base_url='https://hub.nhr.fau.de/api/llmgw/v1', timeout=900)
Q=[{'role':'user','content':'Ein Bauherr errichtet im Außenbereich einen 30 m hohen Mobilfunkmast. '
   'Pruefe knapp die planungsrechtliche Zulaessigkeit nach § 35 BauGB und nenne das Ergebnis.'}]
print(f"{'effort':<10}{'compl':>7}{'reason_tok':>11}{'chars':>7}  reasoning-ish fields")
outs={}
for eff in ['low','medium','high',None]:
    kw={} if eff is None else {'reasoning_effort':eff}
    try:
        r=c.chat.completions.create(model='gpt-oss-120b', messages=Q, seed=1, temperature=0, **kw)
    except Exception as e:
        print(f"{str(eff):<10} ERROR {type(e).__name__}: {str(e)[:120]}"); continue
    m=r.choices[0].message; md=m.model_dump()
    d=getattr(r.usage,'completion_tokens_details',None)
    rt=getattr(d,'reasoning_tokens',None) if d else None
    rc=md.get('reasoning_content') or md.get('reasoning')
    fields=[k for k in md if 'reason' in k.lower() and md[k]]
    outs[str(eff)]=m.content or ''
    print(f"{str(eff):<10}{r.usage.completion_tokens:>7}{str(rt):>11}{len(m.content or ''):>7}  {fields} {'len='+str(len(rc)) if rc else ''}")
print("\nidentical content between efforts?")
ks=list(outs)
for i in range(len(ks)):
    for j in range(i+1,len(ks)):
        print(f"  {ks[i]:>6} vs {ks[j]:<6} identical={outs[ks[i]]==outs[ks[j]]}")
print("\n--- bogus effort value ---")
try:
    r=c.chat.completions.create(model='gpt-oss-120b',messages=Q,reasoning_effort='bananas',seed=1,temperature=0)
    print("  ACCEPTED 'bananas' -> not validated; completion", r.usage.completion_tokens)
except Exception as e:
    print(f"  rejected: {type(e).__name__}: {str(e)[:200]}")
