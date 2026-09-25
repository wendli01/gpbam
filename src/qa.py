import re
from typing import Sequence, Dict, Optional, Union

from src.llm import Generator
from src.prompts import QA_USER, QA_USER_RAG
from src.rag import LawRetriever


class AnswerGenerator(Generator):
    def __init__(self, prompt: str = QA_USER, system_prompt: str = '', model: str = 'meta-llama/llama-3.3-70b-instruct',
                 inference_endpoint: str = None, token_var: str = None,
                 max_tokens: Optional[int] = None, verbose: bool = 0, temperature: Optional[float] = None, name: Optional[str] = None,
                 retriever: LawRetriever = None, **llm_kwargs):

        if name is None:
            name = f"AnswerGenerator ({model.split('/')[-1]})"
        # With a retriever attached, the default prompt would drop the retrieved
        # statutes into an unlabelled slot -- the model is never told what the
        # block is, that it may be irrelevant or incomplete, or that it should
        # still rely on its own knowledge. Swap in the prompt written for that
        # case, unless the caller passed one explicitly.
        if retriever is not None and prompt is QA_USER:
            prompt = QA_USER_RAG
        super().__init__(prompt, system_prompt, model, inference_endpoint, token_var, max_tokens, verbose=verbose,
                         temperature=temperature, name=name, **llm_kwargs)
        self.retriever = retriever

    def build_messages(self, context: str, query: str) -> Sequence[Dict[str, str]]:
        messages = [{'role': 'user', 'content': self.prompt.format(context, query)}]
        if self.system_prompt is not None:
            messages.insert(0, {'role': 'system', 'content': self.system_prompt})
        return messages

    def postprocess(self, response: Optional[str]) -> Optional[str]:
        if response is None:
            return None
        if '<think>' in response.lower():
            response = re.sub(r"(?i)<think>.*?</think>", "", response, flags=re.DOTALL).strip()
        return response

    #: The corpus keys norms by ``jurabk``, which is not how the norm is cited.
    #: Showing "BBauG § 34" for what everyone cites as "§ 34 BauGB" teaches the
    #: model a wrong citation in the same prompt that asks it to cite properly.
    DISPLAY_ALIASES = {
        'BBauG': 'BauGB', 'BayPAG': 'PAG', 'BayVerf': 'BV', 'BayGO': 'GO',
        'BayLStVG': 'LStVG', 'BayAGVwGO': 'AGVwGO', 'BayVfGHG': 'VfGHG',
        'BayLKrO': 'LKrO', 'BayVwZVG': 'VwZVG', 'BayKommZG': 'KommZG',
        'BayKAG': 'KAG', 'BayBezO': 'BezO',
    }

    @classmethod
    def _citation(cls, row) -> str:
        """The norm as a lawyer would cite it: "§ 34 BauGB", "Art. 11 PAG"."""
        book = str(row.get('law_book', '')).strip()
        book = cls.DISPLAY_ALIASES.get(book, book)
        book = re.sub(r'\s+(?:19|20)\d\d$', '', book)      # "StVO 2013" -> "StVO"
        para = str(row.get('paragraph', '')).strip()
        return f'{para} {book}'.strip() if para else book

    @staticmethod
    def _jurisdiction(row) -> str:
        book = str(row.get('law_book', ''))
        return 'Landesrecht Bayern' if book.startswith('Bay') else 'Bundesrecht'

    @staticmethod
    def _heading(row) -> str:
        """The official heading, with the citation prefix the corpus repeats stripped."""
        title = str(row.get('title', '')).strip()
        return title.split(':', 1)[1].strip() if ':' in title else title

    def format_context(self, results: Sequence[Dict[str, str]]) -> str:
        """One XML element per norm, carrying its citation, jurisdiction and heading.

        Attributes rather than a prose header so the metadata cannot be mistaken
        for norm text, and no running index: the prompt asks for conventional
        citations, so a position number is noise the model might echo.
        """
        blocks = []
        for row in results:
            blocks.append(
                f'<norm zitat="{self._citation(row)}" '
                f'gebiet="{self._jurisdiction(row)}" '
                f'ueberschrift="{self._heading(row)}">\n'
                f'{str(row.get("text", "")).strip()}\n'
                f'</norm>')
        return "\n".join(blocks)

    def predict(self, X: Sequence[str], return_raw: bool = False) -> Union[Sequence[str], Sequence[Dict[str, str]]]:
        if self.retriever:
            contexts = self.retriever.predict(X)
            contexts = [f'<context>\n{self.format_context(ctx)}\n</context>' for ctx in contexts]
        else:
            contexts = [''] * len(X)

        message_list = [self.build_messages(ctx, query) for ctx, query in zip(contexts, X)]
        responses = self._predict(message_list)
        # _call_with_retry returns None when a request is rejected outright or
        # exhausts its retries; keep the row and record the failure rather than
        # crashing the whole model's run on one bad response.
        results = [self.postprocess(res['message']) if res is not None else None
                   for res in responses]
        if return_raw:
            for messages, res in zip(message_list, responses):
                if res is not None:
                    res['prompt'] = messages
        return (results, responses) if return_raw else results


class ReWriter(AnswerGenerator):
    def __init__(self, model: str = 'qwen36-35b',
                 prompt: str = 'Extrahiere die  3-5 wichtigsten rechtlichen Konzepte oder Fragestellungen für diesen '
                               'rechtlichen Sachverhalt. Formuliere sie als alleinstehende Suchanfragen ohne jegliche '
                               'andere Präambel oder Erklärung. \n\nSachverhalt:\n{}{}',
                 **llm_kwargs):
        super().__init__(prompt=prompt, model=model, **llm_kwargs)
        self.name = f'ReWriter ({self.name})'


class HyDE(AnswerGenerator):
    def __init__(self, model: str = 'meta-llama/llama-3.3-70b-instruct',
                 prompt: str = 'Erstelle ein kurzes Rechtsgutachten für diesen Sachverhalt. \n\nCase Facts:\n{}{}',
                 **llm_kwargs):
        super().__init__(prompt, model, **llm_kwargs)
        self.name = f'HyDE ({self.name})'
