# Lorenz's csv file structure original

Each CSV row represents one model's generated legal essay for one exam case, together with generation metadata and evaluations from three judge models.

## Files at a glance

| File | Experiment | Rows | Models | Cases per model |
|---|---|---:|---:|---:|
| `experiments/model_study.csv` | Without RAG | 1,458 | 18 | 81 |
| `experiments/archive/zubaers_result_1/essay_writing_WithRag/JUDGE_INSTRUCTION/model_study_JUDGE_INSTRUCTION.csv` | With legal RAG context | 1,377 | 17 | 81 |

The RAG file generally has larger prompts because retrieved legal passages are included in the prompt.

## Essay identity and content

| Column | Meaning |
|---|---|
| `index` | Zero-based exam-case position from `0` to `80`. It repeats for every model, is not a globally unique row ID, and appears only in `model_study.csv`. |
| `model` | Identifier of the model that generated the essay, such as `openai/gpt-5-mini`. |
| `answer` | Final generated legal essay after post-processing, including removal of `<think>...</think>` blocks where present. |
| `message` | Raw response text returned by the generation API. In these two CSVs, it is identical to `answer` in every row. |

## Judge evaluation

Every generated essay is independently evaluated by three judge models:

- `qwen3.5-35b-a3b`
- `deepseek-v3.2`
- `gpt-5-nano`

| Column | Meaning |
|---|---|
| `judgement_Judge (qwen3.5-35b-a3b)` | Complete Qwen judge result, including its written assessment and API metadata. |
| `judgement_Judge (deepseek-v3.2)` | Complete DeepSeek judge result, including its written assessment and API metadata. |
| `judgement_Judge (gpt-5-nano)` | Complete GPT judge result, including its written assessment and API metadata. |
| `score_Judge (qwen3.5-35b-a3b)` | Numeric score extracted from the Qwen judgment. |
| `score_Judge (deepseek-v3.2)` | Numeric score extracted from the DeepSeek judgment. |
| `score_Judge (gpt-5-nano)` | Numeric score extracted from the GPT judgment. |
| `score` | Final ensemble score. It is the minimum of the three judge scores, making it a conservative evaluation. |
| `judging_cost` | Sum of the API costs of all three judging calls for that essay. |

Scores range from `0.0` to `1.0`, where `1.0` represents complete fulfillment of the reference answer. Although the judge instructions request increments of `0.1`, some judge responses contain values such as `0.25` or `0.75`; the score parser accepts these values.

For example, if the three judge scores are `0.75`, `0.9`, and `0.3`, the final `score` is `0.3`.

### Structure of a judgment cell

Each `judgement_Judge (...)` cell is a string representation of a Python dictionary rather than strict JSON. It has approximately this structure:

```python
{
    "message": "Explanation and feedback ... [[0.8]]",
    "prompt_tokens": 18280,
    "completion_tokens": 7996,
    "total_tokens": 26276,
    "time": 51.83,
    "total_cost": 0.01878,
    "temparature": None,
    "top_p": 0.95
}
```

The `message` inside this dictionary contains the judge's explanation, constructive feedback, and final score. The other values describe the judge API call itself.

## Legal-reference similarity

| Column | Meaning |
|---|---|
| `ref_sim` | Legal-reference similarity in `model_study.csv`. |
| `legal_ref_sim` | The same conceptual metric under its current name in the RAG CSV. |

This value is not general semantic or textual similarity. The code extracts unique statutory references as `(law book, section)` pairs from the generated essay and reference answer. It then calculates their Jaccard similarity:

```text
number of shared references / number of all distinct references
```

- `1.0` means both essays contain the same extracted set of statutory references.
- `0.0` means they share no extracted references.
- `0.5` means half of their combined distinct references overlap.

The metric considers the extracted law book and section. It does not directly measure the correctness or quality of the surrounding legal reasoning.

## Answer-generation metadata

These columns describe the API call that generated the essay, not the judge calls.

| Column | Meaning |
|---|---|
| `prompt_tokens` | Provider-reported input-token count. In the RAG experiment, this includes retrieved legal context. |
| `completion_tokens` | Provider-reported output-token count. |
| `total_tokens` | Total provider-reported token usage, normally `prompt_tokens + completion_tokens`. |
| `time` | Duration of the essay-generation request in seconds. |
| `total_cost` | Cost of generating the essay. It does not include `judging_cost`. |
| `temparature` | Misspelling of `temperature` in the implementation. It is blank in every row of both CSVs. |
| `top_p` | Sampling metadata when available from the provider. It is blank for many models. |

## Practical notes

- `total_cost` is the essay-generation cost; `judging_cost` is the combined evaluation cost.
- `answer` and `message` currently duplicate the same essay text, although conceptually `answer` is the post-processed result and `message` is the raw response.
- The RAG CSV has three missing individual judge scores: two DeepSeek scores and one GPT score.
- Essays and judgments contain commas and embedded newlines. Therefore, one CSV record can span many physical text lines. Use `pandas.read_csv()` or Python's `csv` module instead of processing the files one line at a time.
