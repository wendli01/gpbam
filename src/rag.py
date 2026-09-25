import re
from typing import Sequence

import lancedb
import pandas as pd
from lancedb.embeddings import get_registry

from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm


class LawRetriever():
    #: Structural elements the corpus indexes alongside substantive norms --
    #: enacting formulae, tables of contents, annexes. They carry no legal rule
    #: and made up 27.7% of everything the deployed pipeline retrieved.
    STRUCTURAL_RE = re.compile(
        r'Eingangsformel|Inhalts(?:verzeichnis|übersicht)|Schlussformel|Anlage|Anhang|Präambel',
        re.IGNORECASE)

    #: Passages handed to the generator. The published runs used 5, which is thin
    #: for a case whose reference solution cites ~21 distinct norms; raising it is
    #: the cheapest lever on recall. Pass top_k=5 to reproduce the original runs.
    def __init__(self, embedding_model: str = "jinaai/jina-embeddings-v5-text-small-retrieval", top_k: int = 10, hybrid: bool = True,
                 rewriter=None, verbose: bool = 0, kb_name: str = "./my_knowledge_base", device: str = 'cuda',
                 split_queries: bool = True, drop_structural: bool = True, merge: str = 'rrf',
                 candidates: int = 100, reranker: str = None, rerank_length: int = 256,
                 query_type: str = 'hybrid'):
        #: Cross-encoder model name, or None. Multilingual is required -- the
        #: corpus is German. Opt-in because it costs ~12x the retrieval time.
        self.reranker = reranker
        self.rerank_length = rerank_length
        self._reranker_ = None
        self.split_queries = split_queries
        self.drop_structural = drop_structural
        #: How many rows each sub-query contributes to the fusion pool, as
        #: opposed to ``top_k``, which is how many survive it.  Fusion needs
        #: depth: a norm ranked 60th under two sub-queries should outrank one
        #: ranked 5th under a single sub-query, and cannot if the lists were
        #: already cut to top_k.  Truncating at top_k costs 0.86 points of
        #: recall@50.  Also the natural place to over-fetch for a re-ranker.
        self.candidates = candidates
        if merge not in ('rrf', 'roundrobin'):
            raise ValueError(f"merge must be 'rrf' or 'roundrobin', got {merge!r}")
        self.merge = merge
        #: Which retriever the per-sub-query search uses: LanceDB's 'hybrid'
        #: (dense + BM25), 'vector' (dense alone) or 'fts' (BM25 alone).
        #:
        #: Deliberately not the ``hybrid=False`` path below, which is a
        #: different pipeline as well as a different retriever -- it skips
        #: sub-query splitting, structural filtering and the fusion entirely, so
        #: a run through it would differ from a hybrid run in four things at
        #: once. This knob changes the retriever and nothing else, which is what
        #: makes a dense-only arm comparable with the hybrid arms.
        if query_type not in ('hybrid', 'vector', 'fts'):
            raise ValueError(f"query_type must be 'hybrid', 'vector' or 'fts', "
                             f"got {query_type!r}")
        self.query_type = query_type
        self.rewriter = rewriter
        self.top_k = top_k
        self.hybrid = hybrid
        self.verbose = verbose
        self._rewrite_cache = {}  # Cache for re-writing queries
        self.embedding_model = embedding_model
        self.device = device

        self.model_ = self._get_embedding_model()
        ndims = self.model_.ndims()

        self.db_ = lancedb.connect(kb_name)
        self._check_existing_db()

        class Docs(lancedb.pydantic.LanceModel):
            # Core RAG fields
            text: str = self.model_.SourceField()
            vector: lancedb.pydantic.Vector(ndims) = self.model_.VectorField()

            # Metadata promoted to Columns
            title: str
            filename: str
            law_book: str  # e.g., "BGB"
            paragraph: str  # e.g., "§ 2377"
            source_path: str  # Good for debugging local files

        self.schema_ = Docs
        self.model_ = None

    def _get_embedding_model(self):
        return get_registry().get("sentence-transformers").create(name=self.embedding_model, device=self.device)

    def _check_existing_db(self):
        """Checks if the 'documents' table exists and reports status."""
        table_name = "documents"
        if table_name in self.db_.table_names():
            self.table_ = self.db_.open_table(table_name)
            count = len(self.table_)
            if self.verbose:
                print(f"⚠️ Warning: Knowledge base already exists at this location.")
                print(f"-> Loaded existing table '{table_name}' with {count} chunks.")
        else:
            self.table_ = None
            if self.verbose:
                print("✨ New knowledge base initialized (Table 'documents' not found).")

    def fit(self, X: Sequence[dict], batch_size: int = 100, overwrite: bool = False, *_):
        table_name = "documents"

        # Prevent accidental overwrites unless explicit
        if table_name in self.db_.table_names():
            if not overwrite:
                raise ValueError(
                    f"Table '{table_name}' already exists. \nSet `overwrite=True` in fit() if you want to replace it.")
            print(f"⚠️ Warning: Overwriting table '{table_name}' with {len(self.table_)} chunks.'.")

        self.table_ = self.db_.create_table(table_name, schema=self.schema_, mode="overwrite")

        self.model_ = self._get_embedding_model()

        # 2. Add data in batches with a progress bar
        for i in tqdm(range(0, len(X), batch_size), desc="Ingesting to LanceDB"):
            batch = X[i: i + batch_size]

            # If using the Embedding API, the vectorization happens inside here
            self.table_.add(batch)

        self.table_.create_fts_index("text", replace=True)

        self.model_ = None

        if self.verbose:
            print(f"Finished! Total rows in table: {len(self.table_)}")

        return self

    def predict(self, queries: Sequence[str]) -> Sequence[dict]:
        if self.rewriter:
            # TODO move cache to rewriter?
            processed_queries = []
            to_rewrite = []
            indices_to_rewrite = []

            # 1. Check what we already know
            for i, q in enumerate(queries):
                if q in self._rewrite_cache:
                    processed_queries.append(self._rewrite_cache[q])
                else:
                    # Placeholder to maintain order
                    processed_queries.append(None)
                    to_rewrite.append(q)
                    indices_to_rewrite.append(i)

            # 2. Only call the rewriter for new queries
            if to_rewrite:
                if self.verbose > 1:
                    print(f"Rewriting {len(to_rewrite)} new queries...")

                new_rewrites = self.rewriter.predict(to_rewrite)

                # 3. Update cache and fill in the processed_queries list
                for idx, original_idx in enumerate(indices_to_rewrite):
                    rewritten = new_rewrites[idx]
                    self._rewrite_cache[queries[original_idx]] = rewritten
                    processed_queries[original_idx] = rewritten

            queries = processed_queries
            if self.verbose > 1:
                queries_str = '\n\t'.join(queries)
                print(f"Re-written queries: {queries_str}")

        if not self.hybrid:
            self.model_ = self._get_embedding_model()
            # LanceDB's registry returns an EmbeddingFunction wrapper, not a raw
            # SentenceTransformer: it exposes compute_query_embeddings, not encode.
            query_vectors = self.model_.compute_query_embeddings(list(queries))
            self.model_ = None
            batch_results = self.table_.search(query_vectors).limit(self.top_k).to_list()
            return batch_results

        # The rewriter emits several standalone queries per case, one per line.
        # Concatenating them into a single search -- which is what happened
        # before -- dilutes every lexical term and lands the dense vector between
        # topics instead of on any of them. Issue them separately and merge.
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(self._search_case, queries))

        # TODO add re-ranker?

        return results

    def _subqueries(self, query_text: str) -> Sequence[str]:
        if not self.split_queries:
            return [query_text]
        parts = [q.strip(' \t-*0123456789.') for q in str(query_text).splitlines()]
        parts = [q for q in parts if len(q) > 12]
        return parts or [str(query_text)]

    #: Rank constant for reciprocal-rank fusion.  60 is the value from the
    #: original RRF paper and the one the ablation measured; it is large enough
    #: that the top few ranks of a list score similarly, so agreement between
    #: sub-queries matters more than the exact position within any one of them.
    RRF_K = 60

    def _search_case(self, query_text):
        """Search each sub-query separately, then combine the result lists.

        The sub-queries cover different legal questions, so a single productive
        one must not fill the whole budget.  Two ways to prevent that:

        ``rrf``
            Reciprocal-rank fusion.  A norm that several sub-queries agree on
            outranks one that only the luckiest of them found.  Worth +1.62
            points of recall@50 over round-robin in ``recall_battery.py``
            (8.52% against 6.90%), +33% relative at k=10, and the best
            state-law recall of anything measured -- for no extra search.
        ``roundrobin``
            Interleave the lists by rank.  What the published runs used; kept so
            they can be reproduced.
        """
        subs = self._subqueries(query_text)
        if len(subs) == 1:
            rows = self._perform_search(subs[0])
            return self._rerank(query_text, rows) if self.reranker else rows[:self.top_k]
        lists = [self._perform_search(q) for q in subs]
        if self.reranker:
            # Fuse into a pool the re-ranker can reorder, then let it pick. It can
            # only promote what retrieval already found, so the pool has to be
            # deeper than top_k or there is nothing to promote from.
            pool, seen = [], set()
            for lst in lists:
                for r in lst:
                    key = (r.get('law_book'), r.get('paragraph'))
                    if key not in seen:
                        seen.add(key)
                        pool.append(r)
            return self._rerank(' '.join(subs), pool)
        if self.merge == 'rrf':
            score, rows = {}, {}
            for lst in lists:
                for rank, r in enumerate(lst):
                    key = (r.get('law_book'), r.get('paragraph'))
                    score[key] = score.get(key, 0.0) + 1.0 / (self.RRF_K + rank)
                    rows.setdefault(key, r)
            ranked = sorted(score, key=score.get, reverse=True)
            return [rows[k] for k in ranked[:self.top_k]]
        merged, seen = [], set()
        for rank in range(max(len(l) for l in lists)):
            for l in lists:
                if rank < len(l):
                    key = (l[rank].get('law_book'), l[rank].get('paragraph'))
                    if key not in seen:
                        seen.add(key)
                        merged.append(l[rank])
            if len(merged) >= self.top_k:
                break
        return merged[:self.top_k]

    def _get_reranker(self):
        """Load the cross-encoder once, on first use.

        fp32 deliberately: the development card is a Pascal GTX 1080, and GP104
        runs fp16 at 1/64 rate, so the obvious optimisation is threefold slower
        there -- 4 pairs/s against 12.
        """
        if self._reranker_ is None:
            # An object rather than a name means the scorer is already built and
            # is not a local CrossEncoder -- the InnKube-served Qwen3-Reranker-4B
            # is the case this exists for. It is seven times larger than this
            # card can hold and costs no VRAM, so it must not be routed through
            # the loader below.
            if not isinstance(self.reranker, str):
                self._reranker_ = self.reranker
                return self._reranker_
            import torch
            from sentence_transformers import CrossEncoder
            self._reranker_ = CrossEncoder(self.reranker, device=self.device,
                                           max_length=self.rerank_length,
                                           model_kwargs={'dtype': torch.float32})
        return self._reranker_

    def _rerank(self, query_text, rows):
        """Reorder a candidate pool with the cross-encoder and keep ``top_k``.

        Worth 8.52% -> 10.71% recall@50 over reciprocal-rank fusion in
        ``recall_battery.py``, and the best figure measured on every budget --
        but it costs roughly twelve times the retrieval time, so it is opt-in.

        The passage shown leads with ``title``, the citation and official
        heading: that is the part a query naming a norm can match, and it
        survives truncation, which several thousand characters of statute body
        would not.
        """
        if not rows:
            return rows
        model = self._get_reranker()
        pairs = [(query_text, f"{r.get('title', '')}\n{str(r.get('text', ''))[:1200]}")
                 for r in rows]
        scores = model.predict(pairs, batch_size=16, show_progress_bar=False)
        order = sorted(range(len(rows)), key=lambda i: float(scores[i]), reverse=True)
        return [rows[i] for i in order[:self.top_k]]

    def _perform_search(self, query_text):
        # The full-text half is parsed as a query language: parentheses, colons
        # and bare AND/OR/NOT are syntax, and raise before any search happens.
        sanitized = re.sub(r'[^\w\u00c4\u00d6\u00dc\u00e4\u00f6\u00fc\u00df\u00a7 ]+', ' ', str(query_text)).lower()
        sanitized = re.sub(r'\s+', ' ', sanitized).strip()
        if not sanitized:
            return []
        # Over-fetch so that dropping structural hits does not shrink the budget.
        depth = self.candidates or self.top_k
        limit = depth * 4 if self.drop_structural else depth
        rows = self.table_.search(sanitized, query_type=self.query_type).limit(
            limit).to_pandas().to_dict('records')
        if self.drop_structural:
            rows = [r for r in rows
                    if not self.STRUCTURAL_RE.search(str(r.get('paragraph', '')))]
        return rows[:depth]
