# Running Experiments — Practical Guide

How to add a model (generation or judge), route it to an endpoint, and run an
experiment. For the pipeline as a whole — and especially for **which score is
actually reported** — see [README.md](README.md).

---

## Experiment flow overview

```
src/config.py            — which models to run (MODELS_DEV, JUDGE_MODELS_DEV)
endpoints.yaml           — which API endpoint each model uses
.env                     — API keys, one per endpoint
experiments/*.ipynb      — the notebooks you actually run
```

Experiments are run from **notebooks**, each defining one entry-point function:

| Notebook | Function | What it does |
|---|---|---|
| [experiments/essay_writing_norag.ipynb](experiments/essay_writing_norag.ipynb) | `ew_norag` | Generate answers with no retrieval, then judge |
| [experiments/essay_writing_lawrag.ipynb](experiments/essay_writing_lawrag.ipynb) | `ew_withrag` | Retrieve statutes, generate, then judge |
| [experiments/eassy_writing_rejudge.ipynb](experiments/eassy_writing_rejudge.ipynb) | `ew_rejudge` | Re-judge existing answers, no regeneration |

`ew_norag` / `ew_withrag`: iterate over the generation models → generate answers
→ score them with a `JudgeEnsemble` built from the judge models → **save the CSV
after every model**, so an interrupted run resumes and re-runs skip completed
models. `ew_rejudge` skips generation and only re-scores existing answers.

> The sibling `experiments/essay_writing_norag.py` / `essay_writing_lawrag.py`
> files hold the same function and call it at import, so a long run can be
> detached in a `screen` session: `cd experiments && python essay_writing_norag.py`.
> Keep them in sync with the notebooks if you edit one.

---

## Step 1 — Pick your models

Open [src/config.py](src/config.py) and edit the two active tuples.

```python
MODELS_DEV = (                       # generation models
    "GaleneAI/Magistral-Small-2509-FP8-Dynamic",
    "mistralai/Mistral-Medium-3.5-128B",
    # 'deepseek/deepseek-v3.2',     # ← uncomment or add here
)

JUDGE_MODELS_DEV = (                 # the judge ensemble (three judges is the convention)
    'openai/gpt-5-nano',                 # OpenRouter
    'Qwen/Qwen3.6-35B-A3B-FP8',          # NHR@FAU
    'deepseek-ai/DeepSeek-V4-Flash',     # NHR@FAU
)
```

Two things that are easy to get wrong:

- **`MODELS_DEV` and `JUDGE_MODELS_DEV` are the only names `src/config.py`
  actually exports.** The `MODELS`, `MODELS_DEV_SM`, `JUDGE_MODELS` and
  `JUDGE_CANDIDATES` catalogues sit inside the `'''...'''` archive block at the
  bottom of the file — they are a string, not importable. You do **not** need to
  register a model there first; just put its ID straight into `MODELS_DEV`.
- **You can skip editing the file entirely.** Every entry point takes
  `models=` / `judge_models=`, which override the config lists for that call:
  `ew_norag(models=["openai/gpt-5-nano"], judge_models=[...])`.

Judge names must be unique — the last path segment of each judge model becomes
the CSV column name (`score_Judge (<name>)`).

---

## Step 2 — Route each model to an endpoint

Models not listed in [endpoints.yaml](endpoints.yaml) fall through to
`default_endpoint` (**OpenRouter**). To send one elsewhere, add an entry.

### Available endpoints

| Name | URL | Key in `.env` |
|---|---|---|
| `openrouter` (default) | `https://openrouter.ai/api/v1` | `openrouter_api` |
| `nhr_fau` | `https://hub.nhr.fau.de/api/llmgw/v1` | `nhr_fau_api` |
| `innkube` | `https://llms.innkube.fim.uni-passau.de` | `innkube_api` |
| `openai_direct` | `https://api.openai.com/v1` | `OPENAI_API_KEY` |
| `anthropic_direct` | `https://api.anthropic.com/v1` | `ANTHROPIC_API_KEY` |
| `deepseek_direct` | `https://api.deepseek.com/v1` | `DEEPSEEK_API_KEY` |
| `google_direct` | `https://generativelanguage.googleapis.com/v1beta/openai` | `GOOGLE_API_KEY` |
| `mistral_direct` | `https://api.mistral.ai/v1` | `MISTRAL_API_KEY` |
| `local_vllm` | `http://localhost:8000/v1` | `VLLM_API_KEY` |

`endpoints.yaml` is the source of truth; if this table and the file disagree,
believe the file.

### Routing a model

```yaml
models:
  "deepseek-ai/DeepSeek-V4-Flash":
    endpoint: nhr_fau
    model_id: "deepseek-ai/DeepSeek-V4-Flash"   # omit if identical to the key
```

- The key must match **exactly** what you put in `MODELS_DEV` / `JUDGE_MODELS_DEV`.
- `model_id` is only needed when the endpoint expects a different identifier
  (e.g. `openai_direct` wants `gpt-5-mini`, not `openai/gpt-5-mini`).
- Leave a model out of `models:` to keep it on OpenRouter.

### Finding the right `model_id`

```python
# from the project root
from src.llm import list_models
for m in list_models("https://hub.nhr.fau.de/api/llmgw/v1", "nhr_fau_api"):
    print(m.id)
```

### Verifying connectivity

Open [quick-checks/apicheck.ipynb](quick-checks/apicheck.ipynb) and run a
`check_model(...)` call:

```python
check_model(model="deepseek-ai/DeepSeek-V4-Flash", provider="nhr_fau")
```

`model=` here is the **endpoint-native** ID (the `model_id` from
`endpoints.yaml`), and `provider=` is the endpoint name. Run this before
committing to a long run.

---

## Step 3 — Set API keys in `.env`

Project-root `.env`, one key per endpoint you use:

```dotenv
# OpenRouter (default — almost always needed)
openrouter_api=sk-or-...

# Institutional endpoints
nhr_fau_api=...
innkube_api=...

# Native APIs (only when routing models directly)
OPENAI_API_KEY=sk-...
DEEPSEEK_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
MISTRAL_API_KEY=...

# Local vLLM (any non-empty string works)
VLLM_API_KEY=EMPTY
```

---

## Step 4 — Run the experiment

Activate the environment (`conda activate gpabam`), then open the notebook and run
its cells **from the `experiments/` directory** — paths inside the functions are
relative to it.

```python
# no retrieval  -> ./zubaers_result/essay_writing/without_rag/ji2/no_rag_ji2_result.csv
df = ew_norag(judge_instruction_name="ji2", require_confirmation=False)

# with statute retrieval -> ./zubaers_result/essay_writing/with_rag/ji2/with_rag_ji2_result.csv
df = ew_withrag(judge_instruction_name="ji2", require_confirmation=False)
```

- `judge_instruction_name` selects the judge prompt (`"ji1"` / `"ji2"` / `"ji3"`)
  and names the default output folder.
- Each call prints a path/instruction summary and gates the run behind typing
  `yes`; pass `require_confirmation=False` to skip that while iterating.
- Results are written after every model. A re-run **skips** models already in the
  CSV unless you pass `allow_overwrite_existing_models=True`.
- `ew_withrag` builds the legal knowledge base on demand; to build it up front
  run `python lawrag_create_knowledgebase.py` from `experiments/`. It embeds on
  `cuda:0` when a GPU is visible — pass `device="cuda:3"` / `"cpu"` to change
  that, and `law_ret=` to reuse an already-built retriever across calls.

To apply a new judge to answers you already have — or fill in a judge that failed
for one model — use `ew_rejudge` instead; it never regenerates answers and always
writes to a **separate** CSV. See [README.md](README.md#re-judge-an-existing-answer-csv).

---

## Checklist for adding a new model

- [ ] Put the model ID in `MODELS_DEV` (generation) and/or `JUDGE_MODELS_DEV` (judging) — or pass it as `models=` / `judge_models=`
- [ ] Decide the endpoint: OpenRouter → done; otherwise add an entry in `endpoints.yaml`
- [ ] For a non-OpenRouter endpoint, confirm the correct `model_id` with `list_models()`
- [ ] Add that endpoint's key to `.env`
- [ ] Run a `check_model()` cell in `quick-checks/apicheck.ipynb`
- [ ] Run the notebook entry point

---

## Key files at a glance

| File | What to edit |
|---|---|
| `src/config.py` | `MODELS_DEV`, `JUDGE_MODELS_DEV` — the active model lists |
| `endpoints.yaml` | Per-model endpoint routing and key variable names |
| `.env` | API keys, one per endpoint |
| `src/prompts.py` | Judge instructions `ji1` / `ji2` / `ji3` |
| `experiments/essay_writing_norag.ipynb` | No-RAG runner (`ew_norag`) |
| `experiments/essay_writing_lawrag.ipynb` | RAG runner (`ew_withrag`) |
| `experiments/eassy_writing_rejudge.ipynb` | Re-judge runner (`ew_rejudge`) |
| `experiments/essay_writing_{norag,lawrag}.py` | Same runners as plain scripts, for detached `screen` runs |
| `experiments/lawrag_create_knowledgebase.py` | Builds the LanceDB law KB |
| `quick-checks/apicheck.ipynb` | Connectivity smoke tests |

---

## Running a local model with vLLM

The project talks to vLLM through its OpenAI-compatible API. The server and the
notebook process must run in the same container for `http://localhost:8000/v1`
to resolve.

### 1. Start the server

```bash
export CUDA_VISIBLE_DEVICES=0
export VLLM_API_KEY="token-abc123"
export MODEL_ID="Qwen/Qwen2.5-1.5B-Instruct"

vllm serve "${MODEL_ID}" \
  --served-model-name qwen25 \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto \
  --api-key "${VLLM_API_KEY}" \
  --generation-config vllm \
  --gpu-memory-utilization 0.90
```

Keep it running. `--served-model-name` (`qwen25`) is the ID the API expects.

### 2. Route a project-level name to it

`endpoints.yaml` already defines `local_vllm` and several local models. The
pattern is:

```yaml
models:
  "Qwen/Qwen2.5-1.5B-Instruct":
    endpoint: local_vllm
    model_id: "qwen25"        # must match --served-model-name
```

Drop `model_id` when the served name already equals the key (as the other
`local_vllm` entries do).

### 3. Set the key

```dotenv
VLLM_API_KEY=token-abc123
```

`.env` is loaded with `override=True`, so this wins over a stale shell export.

### 4. Enable and check

Add the project-level name to `MODELS_DEV` (and to `JUDGE_MODELS_DEV` only if the
local model should score answers), then verify with the served ID:

```python
check_model(model="qwen25", provider="local_vllm")
```

Once that passes, run the notebook normally — the client picks `local_vllm` from
`endpoints.yaml` with no code changes.
