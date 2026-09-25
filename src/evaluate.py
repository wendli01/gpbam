from src.qa import AnswerGenerator
from src.scoring import Judge, JudgeEnsemble, legal_ref_similarity
from src.config import JUDGE_MODELS_DEV
from typing import Optional, Sequence
import numpy as np
import pandas as pd


def _sum_judging_cost(responses) -> Optional[float]:
    """Sum judge costs, preserving unknown or entirely missing totals as None."""
    costs = [
        response.get('total_cost')
        for response in responses.values()
        if response is not None
    ]
    if not costs or any(cost is None for cost in costs):
        return None
    return sum(costs)


def evaluate_model(answer_generator: AnswerGenerator, tasks: Sequence[str], solutions: Sequence[str],
                   judge: Judge = None, verbose: bool = False, model_name:str=None) -> pd.DataFrame:
    if judge is None:
        if verbose:
            print(f'Creating judge ensemble from config using {JUDGE_MODELS_DEV}')
        judge = JudgeEnsemble([Judge(model=jm) for jm in JUDGE_MODELS_DEV])

    answers, info = answer_generator.predict(tasks, return_raw=True)
    # A generation that exhausts its retries comes back as None here -- qa.predict
    # deliberately keeps the row rather than dropping it. Substitute an empty
    # record so the frame stays aligned with `tasks`; letting pandas choke on the
    # None cost a full chunk of finished essays (eight hours of Kimi) because one
    # call in twenty failed at the very end.
    info_df = pd.DataFrame([i if isinstance(i, dict) else {} for i in info]) \
        .rename(columns={'prompt': 'qa_prompt'})

    judgements, all_scores, scores = judge.predict(tasks, answers, solutions, return_raw=True)

    judging_cost = [_sum_judging_cost(j) for j in judgements]
    # judge_prompt is identical across judges for a given row (same task/answer/solution), so pull
    # it from the first judge response that isn't None (a judge call can fail after all retries).
    judge_prompt = [next((res['prompt'] for res in j.values() if res is not None), None) for j in judgements]

    judgement_df, score_df = pd.DataFrame.from_records(judgements), pd.DataFrame.from_records(all_scores)
    judgements_dict = {'judgement_' + col: judgement_df[col] for col in judgement_df.columns}
    judgements_dict.update({
        'judgement_text_' + col: judgement_df[col].apply(lambda r: r['message'] if r else None)
        for col in judgement_df.columns
    })
    judgements_dict.update({
        'judgement_finish_reason_' + col: judgement_df[col].apply(lambda r: r.get('finish_reason') if r else None)
        for col in judgement_df.columns
    })
    scores_dict = {'score_' + col: score_df[col] for col in score_df.columns}
    scores_dict['legal_ref_sim'] = [legal_ref_similarity(a,b) for a,b in zip(answers, solutions)]
    model_dict = {'answer': answers, 'score': scores, 'judging_cost': judging_cost, 'judge_prompt': judge_prompt,
                  **judgements_dict, **scores_dict, **info_df.to_dict(orient='list')}

    model_df = pd.DataFrame(model_dict)
    # gpbam task position (rows are generated in task order), so downstream
    # re-judging can verify each answer lines up with the right task/solution.
    model_df.insert(0, 'index', range(len(model_df)))
    model_df['model'] = answer_generator.name if model_name is None else model_name
    if verbose:
        print(f'\t{answer_generator.name}:\t{round(model_df.score.mean(), 3)}\n---\n')

    return model_df


def rejudge_model(answers: Sequence[str], tasks: Sequence[str], solutions: Sequence[str], judge: Judge,
                   generation_info: Optional[pd.DataFrame] = None, model_name: Optional[str] = None,
                   verbose: bool = False) -> pd.DataFrame:
    """Re-score an already-generated set of answers under a (possibly new) judge, without
    calling an AnswerGenerator again. Mirrors evaluate_model's output schema so results
    from generation and re-judging runs can be concatenated/compared directly.

    `generation_info` is optional per-row generation metadata (e.g. qa_prompt, token
    counts, cost, time) to carry over unchanged from the original generation run.
    """
    judgements, all_scores, scores = judge.predict(tasks, answers, solutions, return_raw=True)

    judging_cost = [_sum_judging_cost(j) for j in judgements]
    judge_prompt = [next((res['prompt'] for res in j.values() if res is not None), None) for j in judgements]

    judgement_df, score_df = pd.DataFrame.from_records(judgements), pd.DataFrame.from_records(all_scores)
    judgements_dict = {'judgement_' + col: judgement_df[col] for col in judgement_df.columns}
    judgements_dict.update({
        'judgement_text_' + col: judgement_df[col].apply(lambda r: r['message'] if r else None)
        for col in judgement_df.columns
    })
    judgements_dict.update({
        'judgement_finish_reason_' + col: judgement_df[col].apply(lambda r: r.get('finish_reason') if r else None)
        for col in judgement_df.columns
    })
    scores_dict = {'score_' + col: score_df[col] for col in score_df.columns}
    scores_dict['legal_ref_sim'] = [legal_ref_similarity(a, b) for a, b in zip(answers, solutions)]

    model_dict = {'answer': answers, 'score': scores, 'judging_cost': judging_cost, 'judge_prompt': judge_prompt,
                  **judgements_dict, **scores_dict}
    if generation_info is not None:
        model_dict.update(generation_info.reset_index(drop=True).to_dict(orient='list'))

    model_df = pd.DataFrame(model_dict)
    # gpbam task position (answers arrive in task order), keeping re-judged
    # outputs self-describing so they can be re-judged again downstream.
    model_df.insert(0, 'index', range(len(model_df)))
    model_df['model'] = model_name
    if verbose:
        print(f'\t{model_name}:\t{round(model_df.score.mean(), 3)}\n---\n')

    return model_df
