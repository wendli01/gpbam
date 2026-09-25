"""Which reasoning_effort values does each served model accept?
Sends a deliberately invalid value; the validation error names the permitted set.
Rejected requests generate no tokens and are not billed."""
import os, re, openai
from dotenv import load_dotenv
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
load_dotenv(ROOT / '.env')

def permitted(client, model):
    try:
        client.chat.completions.create(model=model, messages=[{'role':'user','content':'hi'}],
                                       max_tokens=1, reasoning_effort='__probe__')
        return 'ACCEPTED INVALID VALUE (not validated)'
    except Exception as e:
        s = str(e)
        lits = re.findall(r"'([a-zA-Z]+)'", s)
        known = [l for l in lits if l in ('minimal','none','low','medium','high','xhigh','xxhigh','max','default','auto')]
        seen, order = set(), []
        for l in known:
            if l not in seen: seen.add(l); order.append(l)
        if order: return ' | '.join(order)
        return ('no reasoning param: ' if 'reasoning' not in s.lower() else '') + re.sub(r'\s+',' ', s)[:150]

fau = openai.OpenAI(api_key=os.environ['API_KEY_FAU'],
                    base_url='https://hub.nhr.fau.de/api/llmgw/v1', timeout=120)
print('=== NHR@FAU ===')
for m in sorted(x.id for x in fau.models.list()):
    print(f'  {m:<48} {permitted(fau, m)}')

orr = openai.OpenAI(api_key=os.environ['API_KEY_OR'],
                    base_url='https://openrouter.ai/api/v1', timeout=120)
print('\n=== OpenRouter (candidate judges) ===')
for m in ['openai/gpt-5.6-luna','openai/gpt-5-mini','openai/gpt-5-nano','openai/gpt-5.4-nano',
          'qwen/qwen3.5-397b-a17b','qwen/qwen3.5-122b-a10b','deepseek/deepseek-v3.2',
          'anthropic/claude-opus-5','x-ai/grok-4.6']:
    print(f'  {m:<48} {permitted(orr, m)}')
