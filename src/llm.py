import os
import time
import random
import warnings
from typing import Sequence, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai
from openai import RateLimitError, APIConnectionError, BadRequestError, InternalServerError
from types import SimpleNamespace
from tqdm import tqdm

import requests
from json import JSONDecodeError

from src.config import get_model_endpoint


def _collect_stream(stream):
    """A streamed completion, reassembled into the shape ``create()`` returns."""
    parts, finish, usage, rid = [], None, None, None
    for chunk in stream:
        rid = rid or getattr(chunk, 'id', None)
        if getattr(chunk, 'usage', None) is not None:
            usage = chunk.usage
        for choice in chunk.choices or ():
            if choice.delta is not None and choice.delta.content:
                parts.append(choice.delta.content)
            if choice.finish_reason is not None:
                finish = choice.finish_reason
    message = SimpleNamespace(content=''.join(parts) or None)
    return SimpleNamespace(id=rid, usage=usage,
                           choices=[SimpleNamespace(message=message, finish_reason=finish)])


def _reset_seconds(exc):
    """Seconds until a rate-limit window reopens, if the server says so."""
    headers = getattr(getattr(exc, 'response', None), 'headers', None) or {}
    for key in ('RateLimit-Reset', 'Retry-After'):
        try:
            if headers.get(key) is not None:
                return min(float(headers.get(key)), 3600.0)
        except (TypeError, ValueError):
            pass
    return None


def list_models(inference_endpoint: str = "https://openrouter.ai/api/v1", token_var: str = "openrouter_api"):
    api_key = os.environ[token_var]
    client = openai.OpenAI(api_key=api_key, base_url=inference_endpoint)
    return client.models.list()


class Generator:
    def __init__(self, prompt: str, system_prompt: str = None, model: str = "meta-llama/llama-3.3-70b-instruct",
                 inference_endpoint: str = None, token_var: str = None,
                 max_tokens: Optional[int] = None, max_retries: int = 20, max_concurrency: int = 10, base_backoff: float = 2,
                 verbose: bool = False, temperature: Optional[float] = None, name: Optional[str] = None,
                 timeout: float = 1800, reasoning: bool = True, top_p: Optional[float] = None,
                 reasoning_effort: str = 'high', max_backoff: float = 60,
                 stream: Optional[bool] = None, provider: Optional[dict] = None):
        self.prompt = prompt
        self.system_prompt = system_prompt

        # Resolve endpoint and key from endpoints.yaml unless caller passes them explicitly.
        if inference_endpoint is None or token_var is None:
            resolved_url, resolved_token_var, resolved_model = get_model_endpoint(model)
            self.inference_endpoint = inference_endpoint if inference_endpoint is not None else resolved_url
            self.token_var = token_var if token_var is not None else resolved_token_var
            self.model = resolved_model
        else:
            self.inference_endpoint = inference_endpoint
            self.token_var = token_var
            self.model = model
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.max_concurrency = max_concurrency
        self.base_backoff = base_backoff
        self.max_backoff = max_backoff
        self.verbose = verbose
        self.temperature = temperature
        self.top_p = top_p
        self.timeout = timeout
        self.name = model if name is None else name
        self.reasoning = reasoning
        self.reasoning_effort = reasoning_effort
        #: OpenRouter provider routing, e.g.
        #: {'order': ['DeepInfra'], 'allow_fallbacks': False, 'data_collection': 'deny'}.
        #: Which provider serves an OpenRouter slug is not cosmetic: see the
        #: DeepSeek fallback block in endpoints.yaml, where one provider
        #: reproduces the NHR@FAU judge seat and another refuses 11 of 12
        #: prompts. Ignored for non-OpenRouter endpoints.
        self.provider = provider
        # NHR@FAU's front proxy returns 502 on any response that has sent nothing
        # for 600 s, so a non-streamed 35k-token essay can never arrive there.
        # Streaming keeps bytes flowing; the result is reassembled unchanged.
        self.stream = 'nhr.fau.de' in self.inference_endpoint if stream is None else stream

        api_key = os.environ[self.token_var]
        # max_retries=0: _call_with_retry is the only retry loop. The SDK's own
        # default (2) re-sent every timed-out request, and NHR@FAU's LiteLLM
        # gateway neither cancels an abandoned request nor merges a resend into
        # the one still running -- each resend is a fresh generation. On 09-11
        # that turned every 35k-token oracle essay into a new copy every 300 s.
        self.client = openai.OpenAI(api_key=api_key, base_url=self.inference_endpoint,
                                    max_retries=0)
        self.model_info = self.get_model_info()

    def get_model_info(self):
        resp = requests.get('https://openrouter.ai/api/v1/models')
        if resp.status_code == 200:
            data = resp.json()['data']
            model_info = {d['id']: d for d in data}
            model = 'openai/' + self.model if 'openai' in self.inference_endpoint else self.model
            if model in model_info:
                return model_info[model]
            else:
                warnings.warn('Could not retrieve model information for ' + self.model)
        else:
            return None

    def _call_with_retry(self, messages: Sequence[Dict[str, str]]) -> Optional[Dict[str, str]]:
        for attempt in range(self.max_retries):
            try:
                start = time.time()
                request_kwargs = {
                    'model': self.model,
                    'messages': messages,
                    'timeout': self.timeout,
                    'reasoning_effort': self.reasoning_effort,
                }
                if self.temperature is not None:
                    request_kwargs['temperature'] = self.temperature
                if self.top_p is not None:
                    request_kwargs['top_p'] = self.top_p
                if 'openrouter.ai' in self.inference_endpoint:
                    # OpenRouter's unified reasoning-control param; native provider
                    # APIs (e.g. OpenAI direct) reject this as an unknown parameter.
                    request_kwargs['extra_body'] = {
                        'reasoning': {'enabled': self.reasoning, "effort": self.reasoning_effort, "exclude": False}
                    }
                    if self.provider:
                        request_kwargs['extra_body']['provider'] = self.provider
                if self.max_tokens is not None:
                    if 'api.openai.com' in self.inference_endpoint:
                        # Native OpenAI API deprecated 'max_tokens' for reasoning
                        # models (gpt-5/o-series) in favor of 'max_completion_tokens'.
                        request_kwargs['max_completion_tokens'] = self.max_tokens
                    else:
                        request_kwargs['max_tokens'] = self.max_tokens

                if self.stream:
                    resp = _collect_stream(self.client.chat.completions.create(
                        stream=True, stream_options={'include_usage': True}, **request_kwargs))
                else:
                    resp = self.client.chat.completions.create(**request_kwargs)
                if resp is not None and resp.choices is not None:
                    content = resp.choices[0].message.content
                    if resp.choices[0].finish_reason is None or content is None:
                        raise APIConnectionError(request=None)
                    # Usage is optional in OpenAI-compatible responses. Some
                    # providers return a valid completion with ``usage=None``.
                    # Keep the completion and represent unavailable accounting
                    # data as None rather than failing (or claiming zero cost).
                    usage = resp.usage
                    prompt_tokens = getattr(usage, 'prompt_tokens', None)
                    completion_tokens = getattr(usage, 'completion_tokens', None)
                    total_tokens = getattr(usage, 'total_tokens', None)
                    reported_cost = getattr(usage, 'cost', None)

                    if reported_cost is not None:
                        cost = reported_cost
                    elif self.model_info and prompt_tokens is not None and completion_tokens is not None:
                        cost_info = self.model_info['pricing']
                        cost = float(cost_info['prompt']) * int(prompt_tokens) + float(
                            cost_info['completion']) * int(completion_tokens)
                    else:
                        cost = None

                    res = {'message': content, 'finish_reason': resp.choices[0].finish_reason,
                           'prompt_tokens': prompt_tokens, 'time': time.time() - start,
                           'completion_tokens': completion_tokens, 'total_tokens': total_tokens,
                           'total_cost': cost}

                    effective_parameters = {}
                    if self.model_info:
                        default_parameters = self.model_info['default_parameters']
                        for name, val in [('temperature', self.temperature), ('top_p', self.top_p)]:
                            effective_parameters[name] = val if val is not None else default_parameters.get(name)

                    return {**res, **effective_parameters}

            except BadRequestError as e:
                # A 400 is the server rejecting the request itself -- context length
                # exceeded, unsupported parameter, malformed payload. Retrying cannot
                # change the outcome, and retrying it under the exponential schedule
                # below used to stall a run for days on a single bad request.
                warnings.warn(f'{self.name}: request rejected, not retrying: {str(e)[:200]}')
                return None

            # InternalServerError: a gateway 5xx (FAU's proxy 502s) fails one
            # request, and used to take the whole run down with it
            except (RateLimitError, APIConnectionError, InternalServerError, JSONDecodeError) as exc:
                if self.verbose:
                    print('.', end='')
                if attempt == self.max_retries - 1:
                    return None

                # A server that says when its window reopens is waited out exactly.
                # GWDG counts rejected requests against the same hourly quota, so
                # retrying on the exponential schedule below spent the whole window
                # on 429s within minutes and let almost no generation through.
                reset = _reset_seconds(exc) if isinstance(exc, RateLimitError) else None
                if reset is not None:
                    time.sleep(reset + random.uniform(1, 15))
                    continue

                # Capped exponential backoff. Uncapped, attempt 19 would sleep for
                # roughly twelve days.
                delay = min(self.base_backoff * (2 ** attempt), self.max_backoff)
                delay += random.uniform(0, delay * 0.1)
                time.sleep(delay)

    def _predict(self, message_list: Sequence[Sequence[Dict[str, str]]]) -> Sequence[Optional[Dict[str, str]]]:
        if self.verbose:
            print(f'Executing predict for {len(message_list)} queries')
            start_time = time.time()
        results = [None] * len(message_list)

        with ThreadPoolExecutor(max_workers=self.max_concurrency) as executor:
            future_to_idx = {executor.submit(self._call_with_retry, messages): i
                             for i, messages in enumerate(message_list)}

            for future in tqdm(as_completed(future_to_idx), total=len(message_list),
                               bar_format=self.name + ' {l_bar}{bar:10}{r_bar}{bar:-10b}'):
                idx = future_to_idx[future]
                if self.verbose:
                    print('\t', idx, '\t', round(time.time() - start_time, 1), 's')
                results[idx] = future.result()

        return results
