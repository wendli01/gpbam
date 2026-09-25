import os
import yaml
from pathlib import Path
from typing import Tuple

_ENDPOINTS_FILE = Path(__file__).parent.parent / "endpoints.yaml"
_endpoint_cfg: dict | None = None


def _load() -> dict:
    global _endpoint_cfg
    if _endpoint_cfg is None:
        if _ENDPOINTS_FILE.exists():
            with open(_ENDPOINTS_FILE) as f:
                _endpoint_cfg = yaml.safe_load(f)
        else:
            _endpoint_cfg = {"endpoints": {}, "models": {}, "default_endpoint": "openrouter"}
    return _endpoint_cfg


def get_model_endpoint(model_id: str) -> Tuple[str, str, str]:
    """Return (base_url, token_env, resolved_model_id) for the given model.

    Resolved model_id may differ from model_id when the endpoint config provides
    a model_id override (e.g. native OpenAI API needs "gpt-5-mini" not "openai/gpt-5-mini").
    Falls back to the default_endpoint for models not listed in endpoints.yaml.
    """
    cfg = _load()
    endpoints = cfg.get("endpoints", {})
    models = cfg.get("models") or {}
    default_name = cfg.get("default_endpoint", "openrouter")

    model_entry = models.get(model_id, {})
    endpoint_name = model_entry.get("endpoint", default_name)
    resolved_model_id = model_entry.get("model_id", model_id)

    ep = endpoints.get(endpoint_name)
    if ep is None:
        raise KeyError(
            f"Endpoint '{endpoint_name}' not found in {_ENDPOINTS_FILE}. "
            f"Check your endpoints.yaml."
        )

    return ep["base_url"], ep["token_env"], resolved_model_id


MODELS_DEV = (
    # 'deepseek/deepseek-r1-0528',
    # 'deepseek/deepseek-v3.2',
    # 'deepseek/deepseek-chat-v3-0324',
    # 'openai/gpt-oss-120b',
    # 'qwen/qwen3-32b',
    # 'qwen/qwen3-235b-a22b-thinking-2507',
    # 'qwen/qwen3.5-397b-a17b',
    # 'qwen/qwen3.5-122b-a10b',
    # 'meta-llama/llama-3.1-8b-instruct',
    # 'meta-llama/llama-3.3-70b-instruct',
    # 'meta-llama/llama-4-maverick',
    # 'mistralai/mistral-large-2512',
    # 'openai/gpt-4o-mini-2024-07-18',
    # 'openai/gpt-5-mini',
    # 'google/gemini-2.5-flash-lite',
    # 'anthropic/claude-haiku-4.5',
    # #
    # #  Judge models here.
    # #
    # 'Qwen/Qwen3.6-35B-A3B-FP8', 
    # 'deepseek-ai/DeepSeek-V4-Flash',
    # 'openai/gpt-5-nano',
    # #
    # # bunch of NHR FAU Models
    # #
    "GaleneAI/Magistral-Small-2509-FP8-Dynamic",
    # "Microsoft/Phi-4-mini-instruct", # out of context for law rag.
    # "MiniMaxAI/MiniMax-M3-MXFP8", # Takes ages
    "RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-FP8",
    # "RedHatAI/gemma-4-31B-it-FP8-block", # Takes ages
    # "google/gemma-4-E4B-it", # Takes ages
    "ibm-granite/granite-4.1-3b",
    "mistralai/Mistral-Medium-3.5-128B",
    # "moonshotai/Kimi-K2.6", # Takes ages
    # #
    # # Uni passau models
    # #
    "qwen3-next-80b-a3b-instruct",
    # "qwen36-35b", # Takes ages
    #
    # #
    # # local models via vllm
    # #
    # 'mistralai/Ministral-3-14B-Reasoning-2512',
    # "windprak/open_steuerllm",
    # "utter-project/EuroLLM-22B-Instruct-2512",
    # "google/gemma-4-31B-it",
    #
    #
    )



JUDGE_MODELS_DEV = (
    'openai/gpt-5-nano', # OpenRouter
    # 'qwen36-35b', # from InnKube@Uni Passau
    'Qwen/Qwen3.6-35B-A3B-FP8', # from NHR@FAU
    # 'qwen/qwen3.6-35b-a3b', # from openrouter
    'deepseek-ai/DeepSeek-V4-Flash', # from NHR@FAU
    )




###
### Archive :not used in the main experiment. Kept fo historical reference. Ignore it for the results in the paper.
###
'''
MODELS = ('deepseek/deepseek-r1-0528', 'deepseek/deepseek-v3.2', 'deepseek/deepseek-chat-v3-0324',
          'openai/gpt-oss-120b', 'openai/gpt-oss-20b',
          'qwen/qwen3-32b', 'qwen/qwen3-235b-a22b-thinking-2507', 'qwen/qwen3-next-80b-a3b-thinking', 'qwen/qwen3-max',
          'qwen/qwen3.5-122b-a10b', 'qwen/qwen3.5-397b-a17b',
          'google/gemma-3-27b-it', 'google/gemma-3-12b-it',
          'meta-llama/llama-3.1-8b-instruct', 'meta-llama/llama-3.3-70b-instruct', 'meta-llama/llama-4-maverick',
          'meta-llama/llama-3.1-405b-instruct',
          'mistralai/mistral-large-2512', 'mistralai/mistral-small-3.1-24b-instruct', 'mistralai/mistral-7b-instruct',
          'allenai/olmo-3.1-32b-think', 'allenai/olmo-3.1-32b-instruct',
          'openai/gpt-4o-mini-2024-07-18', 'openai/gpt-4o-2024-11-20', 'openai/gpt-4.1-mini', 'openai/gpt-4.1',
          'openai/gpt-5-nano', 'openai/gpt-5-mini', 'openai/gpt-5', 'openai/gpt-5.1', 'openai/gpt-5.2',
          'google/gemini-2.5-flash-lite',
          'google/gemini-2.5-pro', 'google/gemini-2.5-flash', 'google/gemini-3-pro-preview',
          'anthropic/claude-haiku-4.5',
          'anthropic/claude-opus-4.5', 'anthropic/claude-3.7-sonnet', 'anthropic/claude-sonnet-4.5',
          'minimax/minimax-m2.1', 'minimax/minimax-m2.5'
          )

MODELS_DEV_SM = ('deepseek/deepseek-v3.2',
                 'openai/gpt-oss-120b',
                 'qwen/qwen3-32b',
                 'meta-llama/llama-3.3-70b-instruct',
                 'mistralai/mistral-small-3.1-24b-instruct',
                 'openai/gpt-4o-mini-2024-07-18',
                 'openai/gpt-5-nano',
                 )

JUDGE_MODELS = ('qwen/qwen3.5-397b-a17b', 'deepseek/deepseek-v3.2', 'openai/gpt-5-mini')
# JUDGE_MODELS_DEV = ('qwen/qwen3.5-35b-a3b', 'deepseek/deepseek-v3.2', 'openai/gpt-5-nano') # ORIGINAL JUDGE_MODELS_DEV


JUDGE_CANDIDATES = ('openai/gpt-4o-2024-11-20', 'openai/gpt-5-mini', 'google/gemini-3-flash-preview',
                    'deepseek/deepseek-v3.2', 'qwen/qwen3-235b-a22b-thinking-2507', 'qwen/qwen3.5-35b-a3b',
                    'qwen/qwen3.5-397b-a17b', 'openai/gpt-4o-mini-2024-07-18', 'openai/gpt-5-nano')
'''