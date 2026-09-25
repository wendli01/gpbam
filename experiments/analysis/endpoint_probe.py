"""Is a model's endpoint answering, and how fast?

Exits 0 if a short completion comes back, 1 otherwise, so a shell script can
gate on it. Costs one ~120-token generation, which is nothing next to an 81-case
arm and much less than discovering the same fact by watching a run fail.

The FAU deployments go away without warning, and a 500 from the LiteLLM gateway
arrives in under two seconds, so a supervised retry loop burns its whole budget
in minutes and reports nothing useful. Gate on this instead.

**deepseek-ai/DeepSeek-V4-Flash is a dead name.** The live deployment is
``deepseek-ai/DeepSeek-V4-Flash-0731``; the unsuffixed key survives in
endpoints.yaml only as an alias onto it, for stored CSVs and older code that
still pass the old name. Calling that group directly cannot reach
aquavan2.rrze.uni-erlangen.de:8000 at all. Probing the old name therefore tests
the alias, which is usually what you want; pass -0731 to test the deployment.

    PYTHONPATH=analysis python analysis/endpoint_probe.py deepseek-ai/DeepSeek-V4-Flash-0731
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import openai

from src.config import get_model_endpoint
from oracle_experiment import api_keys

PROMPT = 'Nenne drei Klagearten der VwGO und erklaere jede in einem Satz.'


def probe(name, max_tokens=120, timeout=120):
    api_keys()
    url, token_var, mid = get_model_endpoint(name)
    client = openai.OpenAI(api_key=os.environ[token_var], base_url=url,
                           timeout=timeout)
    t0 = time.time()
    try:
        r = client.chat.completions.create(
            model=mid, max_tokens=max_tokens,
            messages=[{'role': 'user', 'content': PROMPT}])
    except Exception as e:
        dt = time.time() - t0
        print(f'{name}: DOWN after {dt:.2f}s -- {type(e).__name__}: '
              f'{str(e)[:200]}', flush=True)
        return None
    dt = time.time() - t0
    tok = r.usage.completion_tokens
    print(f'{name}: up, {tok} tok in {dt:.2f}s = {tok / dt:.1f} tok/s',
          flush=True)
    return tok / dt


if __name__ == '__main__':
    # A 120-token ping is not a generation test. On 2026-09-11 the DeepSeek
    # deployment answered a ping in 1.4s while every 9,800-token essay queued
    # behind it hit the 600s client timeout, so the gate opened onto an endpoint
    # that could not do the work, and the queue spent 45 minutes producing no
    # .part at all. Pass a token count near the real one to gate on throughput
    # rather than on reachability.
    #
    #     endpoint_probe.py MODEL [max_tokens] [timeout_seconds]
    name = sys.argv[1]
    tok = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    tmo = float(sys.argv[3]) if len(sys.argv) > 3 else 120
    sys.exit(0 if probe(name, max_tokens=tok, timeout=tmo) else 1)
