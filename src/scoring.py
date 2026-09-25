import re
import unicodedata
import math
import warnings
from typing import Sequence, Optional

import numpy as np

from refex.extractor import RefExtractor
from refex.errors import RefExError

from src.llm import Generator
from src.prompts import JUDGE_USER, JUDGE_SYSTEM


def jaccard_similarity(set_a, set_b):
    """
    Computes the Jaccard Similarity coefficient between two sets.
    Score ranges from 0 (no overlap) to 1 (identical sets).
    """
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0

    intersection = set_a.intersection(set_b)
    union = set_a.union(set_b)

    return len(intersection) / len(union)

# Every Unicode space that is not the ASCII one, mapped to it, and the zero-width
# characters dropped. RefEx matches "\u00a7 31 BauGB" with an ASCII space and not with
# the typographic spaces some models put inside citations; gpt-oss-120b averaged 307
# such characters per essay and none of its 601 raw references survived extraction.
_SPACE_TRANSLATION = {c: ' ' for c in range(0x110000)
                      if chr(c) != ' ' and unicodedata.category(chr(c)) == 'Zs'}
_SPACE_TRANSLATION.update({0x200b: None, 0x200c: None, 0x200d: None, 0xfeff: None})


def normalise_spacing(text):
    """ASCII spacing, so citations parse regardless of the space character used."""
    return str(text).translate(_SPACE_TRANSLATION)


def legal_ref_similarity(reference_text, text, scoring=jaccard_similarity):
    def get_refs(markers):
        for marker in markers:
            for ref in marker.get_references():
                # Truthiness, not `is not None`: RefEx emits a degenerate ('', '')
                # reference, and an empty book/section is not a citation. Two texts
                # yielding only that token used to score a perfect 1.0 for agreeing
                # about nothing.
                if ref.book and ref.section:
                    yield (ref.book, ref.section)

    def extract_refs(s):
        # Deliberately not RefExtractor().extract(). That method collects the
        # markers and then hands them to replace_content(), which rewrites the
        # text with [ref=...] spans and raises RefExError if two markers overlap.
        # We discard that rewritten text and keep only the markers -- so the
        # overlap check was guarding a product this function never uses, while
        # costing it every reference in the document. One marker touching a
        # "____________" blank discarded 47 references from a single essay; 12 of
        # the 2,754 essays in the corpus were hit, the reference solutions of
        # three of the 81 cases among them, which made those cases undefined for
        # every model at once.
        #
        # The markers below are the same objects extract() would have returned
        # had it not raised: extraction is unchanged, only the text-rewriting
        # step is skipped. An overlap does mean two markers claim the same span,
        # so a citation can in principle be counted twice; that is a far smaller
        # error than dropping the whole document.
        #
        # A fresh extractor per text: the two calls must not share state.
        rx = RefExtractor()
        try:
            content = rx.remove_markers(normalise_spacing(s))
            markers = []
            if rx.do_law_refs:
                markers += list(rx.extract_law_ref_markers(content, False))
            if rx.do_case_refs:
                markers += list(rx.extract_case_ref_markers(content))
        except RefExError:
            # A genuine extraction failure, as opposed to the overlap check.
            # None in the current corpus, but it is still fatal for the document.
            return set()
        return set(get_refs(markers))

    ref_set = extract_refs(reference_text)
    student_set = extract_refs(text)
    return scoring(ref_set, student_set)

class Judge(Generator):
    def __init__(self, prompt: str = JUDGE_USER, system_prompt=JUDGE_SYSTEM,
                 model: str = 'meta-llama/llama-3.3-70b-instruct',
                 inference_endpoint: str = None, token_var: str = None,
                 max_tokens: Optional[int] = None, score_pattern=r"\[\[(\d+(?:[.,]\d+)?)\]\]", verbose: bool = False,
                 name: str = None, temperature: Optional[float] = None, **llm_kwargs):
        if name is None:
            name = f"Judge ({model.split('/')[-1]})"

        super().__init__(prompt, system_prompt, model, inference_endpoint, token_var, max_tokens, verbose=verbose,
                         temperature=temperature, name=name, **llm_kwargs)
        self.score_pattern = score_pattern

    def build_messages(self, task, answer, solution):
        messages = [{'role': 'user', 'content': self.prompt.format(task, solution, answer)}]
        if self.system_prompt is not None:
            messages.insert(0, {'role': 'system', 'content': self.system_prompt})
        return messages

    def extract_score(self, response: str) -> Optional[float]:
        if response is None:
            warnings.warn('Empty response')
            return None
        candidates = re.findall(self.score_pattern, response)
        if len(candidates) >= 1:
            try:
                return float(candidates[-1].replace(",", "."))
            except ValueError:
                warnings.warn(f'Could not convert {candidates[0]} to float.')
                return
        warnings.warn(f'Could not extract score from \n\t{response[:120]}')

    @staticmethod
    def _missing(answer) -> bool:
        """Whether a row has no essay to score, as opposed to a bad one.

        Callers hand answers over in three shapes: ``read_csv`` gives float
        ``nan`` for a blank cell, ``str(...)`` around the same cell gives the
        literal ``'nan'``, and a cut-off generation can leave an empty string.
        """
        if answer is None or (isinstance(answer, float) and math.isnan(answer)):
            return True
        text = str(answer).strip()
        return text == '' or text.lower() == 'nan'

    def predict(self, tasks: Sequence[str], answers: Sequence[str], solutions: Sequence[str],
                return_raw: bool = False):
        # A row with no answer is a generation that never landed, not an essay
        # worth no marks. Sent to the judge it comes back as a well-formed
        # "[[0]]", and that 0 is indistinguishable downstream from a real zero:
        # ``skipna`` cannot see it, so a truncated arm reads as a collapsed one.
        # On 2026-09-12 the FAU outage cut _gold_cites and _gold_pad off after
        # 60 and 55 of 81 essays; judging all 81 scored the blank tails 0.0 and
        # put both arms ~20 points below the oracle they actually sit level with.
        # Skipping them scores None -> NaN, which every reader already handles.
        live = [i for i, a in enumerate(answers) if not self._missing(a)]
        message_list = [None] * len(answers)
        for i in live:
            message_list[i] = self.build_messages(tasks[i], answers[i], solutions[i])
        if len(live) < len(answers):
            warnings.warn(f'{self.name}: {len(answers) - len(live)} of {len(answers)} rows '
                          f'have no answer; scoring them None rather than 0')
        responses = [None] * len(answers)
        for i, res in zip(live, self._predict([message_list[i] for i in live])):
            responses[i] = res
        scores = [self.extract_score(res['message']) if res is not None else None
                  for res in responses]
        if return_raw:
            for messages, res in zip(message_list, responses):
                if res is not None:
                    res['prompt'] = messages
        return (responses, scores) if return_raw else scores


class JudgeEnsemble():
    def __init__(self, judges: Sequence[Judge], verbose: bool = False, aggregation=np.min):
        self.judges = judges
        self.verbose = verbose
        self.aggregation = aggregation

    def _aggregate(self, all_scores: Sequence[Sequence[float]]) -> float:
        a = np.array(all_scores, dtype=float)
        agg = self.aggregation(np.nan_to_num(a, nan=1.0), axis=0)
        # nan -> 1.0 is the "this seat abstained, do not let it drag the minimum
        # down" convention. With *every* seat abstaining -- a row with no answer
        # to score -- it would instead read as full marks, so those stay NaN.
        return np.where(np.isnan(a).all(axis=0), np.nan, agg)

    def predict(self, tasks: Sequence[str], answers: Sequence[str], solutions: Sequence[str], return_raw: bool = False):
        assert len(tasks) == len(answers) == len(solutions)

        all_responses, all_scores = [], []
        for judge in self.judges:
            responses, scores = judge.predict(tasks, answers, solutions, return_raw=True)
            if self.verbose:
                print(judge.name, np.nan_to_num(np.array(scores, dtype=float), nan=1.0).mean(axis=0))
            all_responses.append(responses)
            all_scores.append(scores)

        agg_scores = self._aggregate(all_scores)
        responses = [{judge.name: all_responses[i][q] for i, judge in enumerate(self.judges)} for q in
                     range(len(answers))]
        scores = [{judge.name: all_scores[i][q] for i, judge in enumerate(self.judges)} for q in range(len(answers))]
        return responses, scores, agg_scores if return_raw else agg_scores
