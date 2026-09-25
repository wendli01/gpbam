"""One-factor-at-a-time recall sweep over the rebuilt corpora.

Every retrieval number in this repo predates ``rebuild_kb_with_titles.py``, and
the conclusion they supported -- that BM25 beat the hybrid search -- was an
artefact of a dense channel embedded with an English-only model over text that
did not contain the norm's own citation.  Both halves of that are fixed, so the
ablations have to be redone before any of them can be read.

Design decisions that keep this affordable:

**Retrieve once, score many.**  ``recall@k`` for any smaller *k* is a prefix of
the ranked list, so each configuration retrieves once at ``K_MAX`` and every
smaller budget falls out by truncation.  The *k* sweep is therefore free.

**Both gold sets, one retrieval.**  ``refex`` (~21 norms/case, the universe
``legal_ref_sim`` and the older ablations use) and ``gold`` (~52, dual section
and article extraction, what ``corpus_comparison.py`` uses) differ only in the
denominator.  Scoring both costs nothing and removes the need to pick one before
the numbers exist.

**One factor at a time.**  A full cross product is thousands of runs.  Each
configuration below changes exactly one knob away from ``BASELINE``, so every row
reads as a delta against it rather than against a different corner of the space.

Queries come from the cached rewrites, so this measures the pipeline that
actually ran rather than a fresh sample of the rewriter.

Run from ``experiments/``::

    PYTHONPATH=analysis python analysis/recall_battery.py
    PYTHONPATH=analysis python analysis/recall_battery.py --only baseline hybrid_fts
    PYTHONPATH=analysis python analysis/recall_battery.py --table   # re-print, no retrieval
"""

import argparse
import json
import logging
import os
import re
import sys
import time

# Fragmentation is what turns a recoverable allocation failure into a fatal one
# on a card this small, and this script shares it with whatever else is running.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath('..'))

from lancedb.embeddings import get_registry

import refs as refs_mod
from retrieval_ablation import Retriever, norm_of, is_structural
from retrieval_diagnostics import refex_norms, GPBAM, OUT

FEDERAL = './my_knowledge_base_titled'
COMBINED = './my_knowledge_base_bayern_titled'
#: 38,966 Leitsätze from the Bavarian public-law courts, built by
#: ``leitsaetze.py``.  Searched instead of the statute corpus in the ``ls_*``
#: variants; its rows are not norms, so they are never scored directly -- they
#: are resolved into statute rows through their Normenkette first.
LEITSATZ = './my_leitsatz_index'
REWRITES = f'{OUT}/rewrites_qwen3.6-35b-a3b-fp8.json'
REWRITER_MODEL = 'Qwen/Qwen3.6-35B-A3B-FP8'

#: Alternative rewriter prompts, each cached to its own file.
#:
#: The deployed prompt asks for "die wichtigsten rechtlichen Konzepte" -- and the
#: rubric ceiling, which is nothing *but* doctrinal concepts, came last at 2.86%.
#: Concept vocabulary is the one thing statute text reliably does not contain:
#: Art. 42 II VwGO never says *Klagebefugnis*.  Both alternatives below stop
#: asking for concepts and ask for something the corpus actually holds.
REWRITE_PROMPTS = {
    # The corpus is keyed by citation, and the rebuild put that citation into
    # both the vector and the FTS index -- so a query that *is* a citation can
    # now match a norm directly.  This turns the rewriter into a read-out of the
    # model's parametric legal knowledge, which citation_lookup.py showed is
    # substantial even in the no-retrieval essays.
    'citations':
        'Nenne die Rechtsnormen, die für die Lösung dieses Sachverhalts einschlägig '
        'sind. Gib pro Zeile genau eine Norm im üblichen Zitierformat an, zum '
        'Beispiel "§ 34 BauGB" oder "Art. 42 Abs. 2 VwGO". Nenne auch bayerisches '
        'Landesrecht, wo es einschlägig ist. Keine Erklärung, keine Präambel.'
        '\n\nSachverhalt:\n{}{}',
    # HyDE in one call: match statute text with statute-shaped text instead of
    # with the question.  Abstract, in Normsprache, with the case's names and
    # places removed -- the dense channel is good at this and it is exactly the
    # register the index is written in.
    'statute':
        'Formuliere für die 3-5 zentralen Rechtsfragen dieses Sachverhalts jeweils '
        'einen Satz so, wie er im Wortlaut eines deutschen Gesetzes stehen könnte: '
        'abstrakt, in Normsprache, ohne Namen, Orte oder Einzelheiten des Falls. '
        'Eine Formulierung pro Zeile, keine Präambel.'
        '\n\nSachverhalt:\n{}{}',
    # HyDE aimed at the Leitsatz index rather than at statute text.  Every
    # prompt above was written for a corpus of statutes, where the deployed
    # "rechtliche Konzepte" phrasing is fighting the corpus: a statute never
    # names the doctrine it embodies, which is why the rubric ceiling -- pure
    # doctrine -- came last at 2.86%.  A Leitsatz is exactly that missing
    # register, so it should be searched with the register it is written in: an
    # abstract rule sentence, not a topic and not a norm text.
    'leitsatz':
        'Formuliere für die 3-5 zentralen Rechtsfragen dieses Sachverhalts jeweils '
        'einen Leitsatz, wie ihn ein deutsches Gericht einer Entscheidung '
        'voranstellen würde: ein abstrakter Rechtssatz in einem Satz, mit den '
        'juristischen Fachbegriffen, die der Fall aufwirft, aber ohne Namen, Orte '
        'oder Einzelheiten des Sachverhalts. Ein Leitsatz pro Zeile, keine Präambel.'
        '\n\nSachverhalt:\n{}{}',
}
REWRITE_CACHE = {name: f'{OUT}/rewrites_{name}.json' for name in REWRITE_PROMPTS}
RESULTS = f'{OUT}/recall_battery.csv'

K_MAX = 100
K_SWEEP = (5, 10, 20, 50, 100)

#: The deployed pipeline, as ``LawRetriever`` runs it today: rewriter queries,
#: hybrid search, round-robin merge, structural passages dropped.
BASELINE = dict(corpus='combined', mode='hybrid', query='rewriter',
                merge='roundrobin', drop_structural=True, rerank=None,
                fusion_w=None, cite_filter=None, state_share=None, two_hop=None)

#: Cross-encoder for the re-ranking variants.  Multilingual (XLM-R based), which
#: a German corpus needs, and small enough to sit beside the embedder on 8 GB.
#:
#: Run it in **fp32**.  This is a Pascal card and GP104 executes fp16 at 1/64
#: rate, so the obvious optimisation is threefold *slower* here: 4 pairs/s in
#: fp16 against 12 in fp32, and 18 at max_length=256.
#: Measured on this card, fp32, 200 documents of ~350 tokens:
#:
#:   gte-multilingual-reranker-base    3.3 s   60.8 docs/s   1.43 GB   Apache-2.0
#:   bge-reranker-v2-m3                8.7 s   23.1 docs/s   2.30 GB
#:   jina-reranker-v3                 19.7 s   10.2 docs/s   2.48 GB   CC-BY-NC
#:   jina-reranker-v3.5               21.1 s    9.5 docs/s   2.37 GB   CC-BY-NC
#:
#: The listwise Jina models are the strongest on BEIR (63.2 against bge's 51.8)
#: and they do load here -- sdpa, no flash-attention requirement -- but they are
#: twice as slow on Pascal, because throughput *falls* as more candidates are
#: packed into one context (9.5 docs/s at chunk 2, 5.6 at chunk 16, OOM at 32).
#: Attention is quadratic in the packed context and this card is compute-bound,
#: so the fewer-forward-passes argument inverts and they run effectively
#: pointwise anyway. Not worth wiring in here; revisit on Ampere or newer.
#: All of these load through sentence-transformers' ``CrossEncoder``, so they are
#: swaps rather than integrations.  bge and gte are both 2024 models -- the two we
#: happened to measure first are the *older* generation, not the field.
RERANKERS = {
    'bge': 'BAAI/bge-reranker-v2-m3',                              # BEIR 51.8
    # 2.6x faster and half the VRAM, but only ~85M non-embedding parameters, and
    # it shows: second on every budget here and it promotes no state law at all.
    'gte': 'Alibaba-NLP/gte-multilingual-reranker-base',
    # Causal LLM scoring yes/no logits rather than a classic cross-encoder.
    # Apache-2.0, 32k context, 100+ languages, MMTEB-R 66.36. Also the only one
    # that takes a task instruction, which is worth 1-5% on its own numbers and
    # plausibly more on a domain this far from web search.
    'qwen3': 'Qwen/Qwen3-Reranker-0.6B',
    # Apache-2.0, trained with GRPO plus contrastive and preference learning.
    # BEIR 55.57, ahead of bge's 51.8.
    'mxbai': 'mixedbread-ai/mxbai-rerank-base-v2',
    # The pointwise predecessor of the listwise v3/v3.5 -- same family, none of
    # the packing penalty that makes those two slow on Pascal.
    'jinav2': 'jinaai/jina-reranker-v2-base-multilingual',
    # Served by InnKube, not loaded locally -- see InnkubeReranker. 4B, i.e. ~7x
    # anything this card can hold, and it costs no VRAM at all.
    'innkube': 'qwen3-reranker-4b',
}

#: Re-rankers that run off-box, so they need no GPU headroom and no unloading.
REMOTE_RERANKERS = {'innkube'}
RERANKER = RERANKERS['bge']
#: Leitsätze consulted per case in the two-hop.  Each names ~4 distinct norms
#: after de-duplication, so 20 is roughly what it takes to fill a budget of 100
#: from chains alone; 50 is measured as well, to find where following one more
#: authority stops paying.
LS_DEPTH = 20

RERANK_POOL = 200      # candidates fused before re-ranking
RERANK_LEN = 256       # tokens per pair; the citation and heading lead the text

#: Each entry overrides exactly one key of BASELINE.  ``ceiling`` variants are
#: not achievable configurations -- they bound what a better query could buy.
VARIANTS = {
    'baseline': {},
    # --- which corpus
    'corpus_federal': dict(corpus='federal'),
    # The corpus contrast is printed next to a ladder whose columns start at
    # RRF -- round-robin is in COLS but not in PAPER_COLS. Holding the federal
    # side at BASELINE's round-robin would measure the two corpora at a fusion
    # the reader never sees, so the corpus table uses this one.
    'corpus_federal_rrf': dict(corpus='federal', merge='rrf'),
    # --- the hybrid question: is the dense channel worth anything now?
    'mode_fts': dict(mode='fts'),
    'mode_vector': dict(mode='vector'),
    # The retriever the `dense` *generation* arm actually uses. mode_vector
    # above inherits BASELINE's round-robin merge, so its recall describes a
    # different pipeline than the essays do, and a paper table that puts the two
    # in one column would be comparing two things. Same for bm25.
    'dense_rrf': dict(mode='vector', merge='rrf'),
    'bm25_rrf': dict(mode='fts', merge='rrf'),
    # --- what we search with
    'query_facts': dict(query='facts'),
    'query_sentences': dict(query='sentences'),
    'query_paragraphs': dict(query='paragraphs'),
    'query_rubric_CEILING': dict(query='rubric'),
    'query_solution_CEILING': dict(query='solution'),
    # --- alternative rewriter prompts, same model, same everything else
    'rw_citations': dict(query='rw_citations'),
    'rw_statute': dict(query='rw_statute'),
    # --- how the sub-query result lists are combined, at the same total budget
    'merge_concat': dict(merge='concat'),
    'merge_rrf': dict(merge='rrf'),
    # --- re-ranking: fuse a deep pool, then reorder it with a cross-encoder
    'rerank_bge': dict(merge='rrf', rerank='bge'),
    # How deep a pool the re-ranker gets to reorder. It can only promote what
    # retrieval already found, and recall@100 was still climbing at 14.21%, so
    # there should be gold below rank 100 for a deeper pool to reach.
    'rerank_pool100': dict(merge='rrf', rerank='bge', pool=100),
    'rerank_pool400': dict(merge='rrf', rerank='bge', pool=400),
    # 2.6x faster than bge and Apache-2.0; the question is whether it ranks well
    # enough on German statute text to be worth switching to.
    'rerank_gte': dict(merge='rrf', rerank='gte'),
    'rerank_innkube': dict(merge='rrf', rerank='innkube'),
    'rerank_qwen3': dict(merge='rrf', rerank='qwen3'),
    'rerank_mxbai': dict(merge='rrf', rerank='mxbai'),
    'rerank_jinav2': dict(merge='rrf', rerank='jinav2'),

    # --- citation as a key, not as a query
    # rw_citations was the worst configuration measured, because "§ 34 BauGB" is
    # a poor *similarity* query: the dense channel has almost nothing to embed
    # and BM25 tokenises it into §, 34, baugb, where § and 34 match thousands of
    # norms. The norms the model names are still worth having -- they just want
    # an exact lookup on (law_book, paragraph) instead of a search.
    'cite_only': dict(cite_filter='only'),
    'cite_then_search': dict(merge='rrf', cite_filter='first'),

    # --- the two-hop: search Leitsätze, follow their Normenketten to the norms
    # The doctrinal register the statute corpus does not contain.  ``ls_only``
    # isolates the channel; ``ls_then_search`` is the deployable shape; the last
    # one asks whether the model's own citations and a court's chains are
    # redundant or complementary.
    'ls_only': dict(two_hop='only'),
    # What to search the Leitsätze *with*.  ``ls_only`` runs no statute search,
    # so the query reaches the two-hop and nothing else, which is what makes
    # this a clean read on the rewriter prompt.  Worth redoing here because
    # every earlier query ablation was measured against statute text: the
    # ordering there (concepts > statute-style > rubric) may well invert on a
    # corpus that is written in doctrinal language rather than in Normsprache.
    'ls_q_leitsatz': dict(two_hop='only', query='rw_leitsatz'),
    'ls_q_statute': dict(two_hop='only', query='rw_statute'),
    'ls_q_rubric_CEILING': dict(two_hop='only', query='rubric'),
    'ls_q_facts': dict(two_hop='only', query='facts'),
    'ls_then_search': dict(merge='rrf', two_hop='first'),
    'ls_deep50': dict(merge='rrf', two_hop='first', ls_depth=50),
    'ls_shallow10': dict(merge='rrf', two_hop='first', ls_depth=10),
    # The redistributable subset -- 1,588 Leitsätze the court wrote itself, which
    # § 5 Abs. 1 UrhG puts outside copyright.  Its reach ceiling is 35.2% against
    # 60.1% for all of them, so this measures what a shippable index would cost.
    'ls_amtlich': dict(merge='rrf', two_hop='first', ls_kinds=('amtlich',)),
    'cite_ls_search': dict(merge='rrf', cite_filter='first', two_hop='first'),

    # --- jurisdiction routing
    # 40.7% of gold citations are Bavarian state law, but state law is a small
    # minority of the corpus and loses on similarity to the far larger federal
    # body. These reserve a share of the budget for it.
    'state_share40': dict(merge='rrf', state_share=0.4),
    'state_share60': dict(merge='rrf', state_share=0.6),

    # --- how the dense and lexical channels are weighted against each other
    # LanceDB's hybrid search fuses them with an untuned default. Dense alone
    # already beats BM25 alone roughly two to one here, so the balance is
    # probably not where the default puts it. 1.0 is dense-only, 0.0 lexical.
    'fusion_w25': dict(merge='rrf', fusion_w=0.25),
    'fusion_w50': dict(merge='rrf', fusion_w=0.50),
    'fusion_w75': dict(merge='rrf', fusion_w=0.75),
    'fusion_w90': dict(merge='rrf', fusion_w=0.90),
    # --- is the structural filter earning its place?
    'nostructural': dict(drop_structural=False),
}


# --------------------------------------------------------------------------
# queries

def free_gpu():
    """Give the card back what the previous retriever was holding."""
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def gpu_free_mib():
    """Free VRAM right now, or ``None`` if there is no card to ask."""
    import subprocess
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=memory.free',
                              '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=10)
        return int(out.stdout.split()[0])
    except Exception:
        return None


def wait_for_gpu(need_mib, timeout=3600, every=60):
    """Block until the card has ``need_mib`` free, rather than racing for it.

    This shares an 8 GB card with the generation runs, and losing a two-hour
    sweep to a transient overlap is worse than waiting a few minutes for one.
    Proceeds anyway on timeout, so a permanently busy card fails loudly at the
    allocation instead of hanging forever.
    """
    waited = 0
    while waited < timeout:
        free = gpu_free_mib()
        if free is None or free >= need_mib:
            return
        print(f'    waiting for GPU: {free} MiB free, need {need_mib} '
              f'({waited // 60} min so far)', flush=True)
        time.sleep(every)
        waited += every
    print(f'    proceeding after waiting {timeout // 60} min for the GPU', flush=True)


def _clean_lines(text, min_chars=12, max_q=8):
    out, seen = [], set()
    for l in str(text).splitlines():
        l = re.sub(r'\s+', ' ', l.strip(' #*-–—\t0123456789.')).strip()
        if len(l) > min_chars and l.lower() not in seen:
            seen.add(l.lower())
            out.append(l)
    return out[:max_q]


def unwrap(text):
    """Undo the source's hard wrap, including the words it breaks in half.

    The facts and solutions are wrapped at ~76 columns and the wrap splits words
    mid-token -- ``kreisange-\\nhörigen``, ``Nach-\\nbargrundstücke``.  Replacing
    the newline with a space, which is what every query builder here used to do,
    turns one word into two unusable fragments: 2,041 of them across the facts
    and 14,592 across the solutions.  A trailing hyphen before a lower-case
    letter is a wrap break and the hyphen goes; before an upper-case letter it is
    a real compound hyphen and only the newline goes.
    """
    t = str(text)
    t = re.sub(r'-\n\s*(?=[a-zäöüß])', '', t)
    t = re.sub(r'-\n\s*(?=[A-ZÄÖÜ])', '-', t)
    return re.sub(r'\s+', ' ', re.sub(r'\s*\n\s*', ' ', t)).strip()


def make_rewrites(facts, name, model=REWRITER_MODEL):
    """Rewrite every case under an alternative prompt, cached to disk.

    One call per case on a free endpoint.  Cached because the rewriter samples,
    and two variants compared on different samples would not be a comparison of
    prompts.
    """
    path = REWRITE_CACHE[name]
    if os.path.exists(path):
        out = json.load(open(path))
        if len(out) == len(facts):
            return out
    from dotenv import load_dotenv
    load_dotenv('../.env', override=True)
    for env_name, yaml_name in (('API_KEY_UP', 'innkube_api'),
                                ('API_KEY_FAU', 'nhr_fau_api'),
                                ('API_KEY', 'openrouter_api')):
        if env_name in os.environ:
            os.environ.setdefault(yaml_name, os.environ[env_name])
    from src import qa
    print(f'rewriting {len(facts)} cases with prompt "{name}" ({model}) ...', flush=True)
    out = [o or '' for o in
           qa.ReWriter(model=model, prompt=REWRITE_PROMPTS[name]).predict(list(facts))]
    json.dump(out, open(path, 'w'), ensure_ascii=False, indent=1)
    n = [len(_clean_lines(o)) for o in out]
    print(f'  -> {path}: {np.mean(n):.1f} queries per case '
          f'(min {min(n)}, max {max(n)}), {sum(1 for o in out if not o)} failed', flush=True)
    return out


def build_queries(facts, sols, rubric, rewrites, alt_rewrites=None):
    """Every query variant, precomputed once per case."""
    #: strip citations from the rubric so the ceiling measures the *concepts*, not
    #: a lookup of the answer's own references
    strip = re.compile(r'(?:§+|Art\.?)\s*[\d IVXAbslt.,;()§-]*|'
                       r'\b[A-ZÄÖÜ][A-Za-zÄÖÜäöüß]{2,}[GO]\b')
    q = {}
    for i, f in enumerate(facts):
        text = unwrap(f)
        sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+(?=[A-ZÄÖÜ])', text)
                 if len(s.strip()) > 20]
        per = max(1, len(sents) // 6)
        q[i] = {
            'facts': [text],
            'sentences': [' '.join(sents[j:j + per]) for j in range(0, len(sents), per)],
            # None, not a fallback to the full text: the facts carry no
            # blank-line structure at all (0 of 81 cases), so falling back
            # silently recorded a copy of `facts` under a second name and made
            # it look like an independent measurement.  The same fallback is in
            # retrieval_ablation.q_paragraphs, so its stored "paragraphs" rows
            # are duplicates too.
            'paragraphs': (lambda parts: [unwrap(p) for p in parts] if len(parts) > 1 else None)(
                [p for p in re.split(r'\n\s*\n', str(f)) if len(p.strip()) > 120]),
            'rewriter': _clean_lines(rewrites[i]) or [text],
            'rubric': _clean_lines(strip.sub(' ', str(rubric.get(i, '')))) or [text],
            'solution': [unwrap(sols[i])],
        }
        for name, texts in (alt_rewrites or {}).items():
            # A citation is short: "§ 34 BauGB" is ten characters, under the
            # 12-character floor the prose variants need to drop bullet
            # fragments.  Applying that floor here deleted the very queries the
            # citations prompt exists to produce.
            # No cap on citations. max_q=8 exists to stop the prose variants
            # issuing twenty searches per case; an exact lookup costs nothing per
            # extra citation, and capping it threw away 40% of the channel's
            # gold norms (276 recovered instead of 458).
            q[i][f'rw_{name}'] = _clean_lines(
                texts[i], min_chars=4 if name == 'citations' else 12,
                max_q=10 ** 6 if name == 'citations' else 8) or None
    return q


# --------------------------------------------------------------------------
# retrieval

def merge_lists(lists, k, how):
    """Combine per-sub-query result lists into one ranked list of ``k`` norms.

    All three policies spend the *same* total budget, so the comparison is about
    allocation and not about handing one variant more passages.  Giving each
    sub-query its own budget of ``k`` is not a merge policy but a larger *k*,
    which the sweep already covers.
    """
    merged, seen = [], set()
    if how == 'concat':                      # a single search already happened
        lists = [lists[0]]
    if how == 'rrf':
        # reciprocal rank fusion across sub-queries: a norm that several
        # sub-queries agree on outranks one that only the luckiest found
        score = {}
        rows = {}
        for lst in lists:
            for rank, r in enumerate(lst):
                key = (r.get('law_book'), r.get('paragraph'))
                score[key] = score.get(key, 0.0) + 1.0 / (60 + rank)
                rows.setdefault(key, r)
        for key in sorted(score, key=score.get, reverse=True)[:k]:
            merged.append(rows[key])
        return merged
    for rank in range(max((len(l) for l in lists), default=0)):
        for lst in lists:
            if rank < len(lst):
                key = (lst[rank].get('law_book'), lst[rank].get('paragraph'))
                if key not in seen:
                    seen.add(key)
                    merged.append(lst[rank])
        if len(merged) >= k:
            break
    return merged[:k]


#: What a candidate row needs to carry between the two phases. The search result
#: also holds the 1024-float vector, which is 66 MB across 81 cases x 200 rows
#: and useless downstream.
POOL_FIELDS = ('text', 'title', 'law_book', 'paragraph', 'source_path')


def slim(row):
    return {k: row.get(k) for k in POOL_FIELDS}


def unload_query_embedder(ret):
    """Move LanceDB's query embedder off the GPU, keeping the model usable.

    Retrieval and re-ranking are sequential per case, but looping case-by-case
    keeps both models resident for the whole run -- and on an 8 GB card shared
    with a generation job, embedder (2.4 GB) plus a 0.6B re-ranker (2.4 GB) does
    not fit. Retrieving every pool first and only then loading the re-ranker
    makes the peak ``max(embedder, reranker)`` instead of their sum.

    The model is reached through the table's registered embedding function
    rather than the Retriever, because that is where LanceDB actually holds it;
    dropping the Retriever frees nothing. Moving it to CPU rather than deleting
    it keeps a later configuration able to move it back.
    """
    freed = []
    try:
        for name, c in get_registry().parse_functions(ret.table.schema.metadata).items():
            m = c.function.get_embedding_model()
            m.to('cpu')
            freed.append(name)
    except Exception as e:
        print(f'    could not unload the query embedder ({type(e).__name__}: {e})',
              flush=True)
    free_gpu()
    return freed


#: What Qwen3-Reranker is told the task is. It is the only re-ranker here that
#: takes an instruction, and its own numbers put that at 1-5% -- measured against
#: generic web retrieval, so plausibly more on a domain this far from it.
QWEN3_INSTRUCT = ('Given a German public-law exam case, retrieve the statutory '
                  'norms that govern it')


class Qwen3Reranker:
    """Qwen3-Reranker through its native scoring path.

    It is a causal LM, not a sequence classifier: relevance is the probability
    it assigns to answering "yes" rather than "no" at the final position. Wrapping
    it in a classification head produces random scores (see ``load_reranker``),
    so the prompt template and the yes/no logit read-out are reproduced here.

    Exposes ``predict(pairs, batch_size=...)`` so it is interchangeable with a
    ``CrossEncoder`` from ``apply_rerank``'s point of view.
    """

    PREFIX = ('<|im_start|>system\nJudge whether the Document meets the requirements '
              'based on the Query and the Instruct provided. Note that the answer can '
              'only be "yes" or "no".<|im_end|>\n<|im_start|>user\n')
    SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'

    def __init__(self, name, device='cuda', max_length=RERANK_LEN,
                 instruction=QWEN3_INSTRUCT):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.torch = torch
        self.device, self.max_length, self.instruction = device, max_length, instruction
        self.tok = AutoTokenizer.from_pretrained(name, padding_side='left')
        # device_map streams the shards straight onto the card. Materialising the
        # fp32 weights first and then calling .to(device) needs the 2.4 GB twice
        # over at the moment of the copy, which does not fit beside a generation
        # job on 8 GB -- it OOMs asking for 2.31 GiB with 1.52 GiB free.
        self.model = AutoModelForCausalLM.from_pretrained(
            name, dtype=torch.float32, low_cpu_mem_usage=True,
            device_map={'': device}).eval()
        self.yes = self.tok.convert_tokens_to_ids('yes')
        self.no = self.tok.convert_tokens_to_ids('no')
        self.pre = self.tok.encode(self.PREFIX, add_special_tokens=False)
        self.suf = self.tok.encode(self.SUFFIX, add_special_tokens=False)

    def predict(self, pairs, batch_size=8, show_progress_bar=False):
        torch = self.torch
        out = []
        for i in range(0, len(pairs), batch_size):
            chunk = pairs[i:i + batch_size]
            texts = [f'<Instruct>: {self.instruction}\n<Query>: {q}\n<Document>: {d}'
                     for q, d in chunk]
            enc = self.tok(texts, add_special_tokens=False, truncation=True,
                           max_length=self.max_length - len(self.pre) - len(self.suf))
            ids = [self.pre + e + self.suf for e in enc['input_ids']]
            batch = self.tok.pad({'input_ids': ids}, padding=True,
                                 return_tensors='pt').to(self.device)
            with torch.no_grad():
                logits = self.model(**batch).logits[:, -1, :]
            # relevance = P(yes) against P(no), the two tokens the prompt allows
            two = torch.stack([logits[:, self.no], logits[:, self.yes]], dim=1)
            out.extend(torch.log_softmax(two, dim=1)[:, 1].exp().float().cpu().tolist())
        return out


class InnkubeReranker:
    """Qwen3-Reranker-4B over InnKube's ``/v1/rerank`` endpoint.

    The best re-ranking option available to this project, and the cheapest: it is
    seven times larger than anything the local 8 GB card can hold, costs no VRAM,
    runs on hardware without Pascal's fp32-only handicap, and is free. It also
    removes the reason the local re-rankers needed the retrieve-then-unload
    dance, since nothing has to share the GPU with the query embedder.

    Same ``predict(pairs, ...)`` interface as a ``CrossEncoder``. The endpoint
    takes one query against many documents, and ``apply_rerank`` only ever passes
    pairs sharing a query, so the pairs are regrouped into one call per chunk.
    """

    def __init__(self, model='qwen3-reranker-4b', chunk=100, timeout=180, retries=4):
        from dotenv import load_dotenv
        load_dotenv('../.env', override=True)
        self.key = os.environ.get('innkube_api') or os.environ['API_KEY_UP']
        self.url = 'https://llms.innkube.fim.uni-passau.de/v1/rerank'
        self.model, self.chunk = model, chunk
        self.timeout, self.retries = timeout, retries

    def _call(self, query, docs):
        import json as _json
        import urllib.request
        body = _json.dumps({'model': self.model, 'query': query,
                            'documents': docs}).encode()
        last = None
        for attempt in range(self.retries):
            try:
                req = urllib.request.Request(
                    self.url, data=body,
                    headers={'Authorization': f'Bearer {self.key}',
                             'Content-Type': 'application/json'})
                d = _json.load(urllib.request.urlopen(req, timeout=self.timeout))
                out = [0.0] * len(docs)
                for r in d['results']:          # returned sorted, so index matters
                    out[int(r['index'])] = float(r['relevance_score'])
                return out
            except Exception as e:              # transient endpoint failures
                last = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f'rerank endpoint failed after {self.retries} tries: {last}')

    def predict(self, pairs, batch_size=None, show_progress_bar=False):
        if not pairs:
            return []
        query = pairs[0][0]
        docs = [d for _, d in pairs]
        out = []
        for i in range(0, len(docs), self.chunk):
            out.extend(self._call(query, docs[i:i + self.chunk]))
        return out


class _CatchInitWarnings(logging.Handler):
    """Collects transformers' 'newly initialized' warnings during a load."""

    def __init__(self):
        super().__init__()
        self.msgs = []

    def emit(self, record):
        self.msgs.append(record.getMessage())


def load_reranker(key='bge', device='cuda'):
    """Load a re-ranker, refusing any whose scoring head is randomly initialised.

    ``CrossEncoder`` will happily wrap a model in a sequence-classification head
    that the checkpoint does not contain. Qwen3-Reranker and mxbai-rerank-v2 are
    both causal LMs that score through yes/no token logits, so loading them this
    way silently produces a ``score.weight`` of random numbers -- and then
    *plausible-looking* scores. Qwen3 rated a Klagebefugnis query 0.3751 against
    the governing VwGO norm and 0.3362 against a Binnenschifffahrt regulation:
    ordered correctly by luck, and meaningless. That is worse than a crash,
    because it would have entered the results table as "this re-ranker does not
    help".

    So the warning transformers emits is escalated to an exception.
    """
    import torch
    from sentence_transformers import CrossEncoder

    if key in REMOTE_RERANKERS:
        return InnkubeReranker(RERANKERS[key])
    # models whose scores only mean anything through their own scoring path
    if key == 'qwen3':
        return Qwen3Reranker(RERANKERS[key], device=device)

    catcher = _CatchInitWarnings()
    log = logging.getLogger('transformers.modeling_utils')
    log.addHandler(catcher)
    try:
        ce = CrossEncoder(RERANKERS[key], device=device, max_length=RERANK_LEN,
                          trust_remote_code=True,
                          model_kwargs={'dtype': torch.float32})
    finally:
        log.removeHandler(catcher)
    bad = [m for m in catcher.msgs if 'newly initialized' in m]
    if bad:
        raise RuntimeError(
            f'{RERANKERS[key]} loaded with randomly initialised weights, so its '
            f'scores would be noise. It needs its native scoring path, not '
            f'CrossEncoder. transformers said: {bad[0][:200]}')
    # The LLM-based re-rankers (Qwen3) are causal models whose tokenizer ships no
    # padding token, so any batch larger than one raises "Cannot handle batch
    # sizes > 1 if no padding token is defined". Encoder re-rankers (bge, gte)
    # already have one and are untouched by this.
    # Two separate gaps, and they do not travel together: Qwen3's *tokenizer*
    # already defines a pad token while its *model config* does not, so guarding
    # the config fix behind the tokenizer check skips it and the sequence
    # classification head still raises "Cannot handle batch sizes > 1 if no
    # padding token is defined". Fix each independently.
    tok = ce.tokenizer
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.padding_side = 'left'          # correct side for a decoder
    if getattr(ce.model.config, 'pad_token_id', None) is None:
        ce.model.config.pad_token_id = tok.pad_token_id
    return ce


def apply_rerank(model, query, rows, top_n):
    """Reorder a fused candidate pool with the cross-encoder.

    The passage shown to the re-ranker leads with ``title`` -- the citation and
    official heading -- because that is the part a query naming a norm can match
    and it survives the 256-token truncation, which several thousand characters
    of statute body would not.
    """
    if not rows:
        return rows
    pairs = [(query, f"{r.get('title', '')}\n{str(r.get('text', ''))[:1200]}") for r in rows]
    scores = model.predict(pairs, batch_size=16, show_progress_bar=False)
    order = np.argsort(np.asarray(scores))[::-1]
    return [rows[i] for i in order[:top_n]]


def citation_index(ret):
    """(law_book, section) -> row, for looking a norm up by its citation.

    Built once per corpus from the table itself, so a citation the model names
    is answered by an exact match rather than by nearest-neighbour search.
    """
    tbl = ret.table.to_lance().to_table(
        columns=['text', 'title', 'law_book', 'paragraph', 'source_path']).to_pylist()
    idx = {}
    for r in tbl:
        n = norm_of(r)
        if n and n not in idx:      # first occurrence wins; duplicates are reprints
            idx[n] = r
    return idx


def leitsatz_rows(ls_ret, idx, subs, depth, mode='hybrid', kinds=None):
    """Statute rows reached by searching the Leitsatz index and following its chains.

    The two-hop.  Hop one searches 38,966 abstracted rule statements -- the only
    text in any corpus written in the register an exam question is written in,
    because a statute never names the situation it governs and a decision buries
    the rule under twenty thousand characters of procedural history.  Hop two
    reads the Normenkette off each retrieved Leitsatz and resolves it against the
    statute corpus by exact key.

    This is ``cite_then_search`` with the keys sourced externally.  That matters
    because ``cite_only`` saturates at 11.16% -- it plateaus from k=20 onward,
    since it can only name norms the model already knows.  A retrieved authority
    has no such ceiling: the reach measurement puts 60.1% of the gold set inside
    the chains of decisions that carry a Leitsatz, against 57.9% for the entire
    decision corpus at 37x the embedding cost.

    Leitsatz rows are never scored as norms themselves -- ``law_book`` holds a
    court and ``paragraph`` a docket number, so ``norm_of`` would be meaningless
    on them.  Only what their chains resolve to enters the ranking.
    """
    lists = []
    for s in subs:
        rows = ls_ret.search(s, depth, mode, drop_structural=False)
        if kinds:
            rows = [r for r in rows if str(r.get('ls_kind')) in kinds]
        lists.append(rows)
    # Fuse by rank before following any chain: a Leitsatz several sub-queries
    # agree on is a better bet than one a single lucky sub-query found, and
    # every chain followed costs budget that cannot be spent on another.
    ranked = merge_lists(lists, depth, 'rrf')
    out, seen = [], set()
    for key in refs_mod.chain_norms_ordered(r.get('norms') or '' for r in ranked):
        if key in idx and key not in seen:
            seen.add(key)
            out.append(idx[key])
    return out


def cited_rows(idx, lines):
    """Rows for every citation the rewriter named that the corpus actually holds.

    Extraction runs over the whole block rather than line by line, because
    ``refs.extract`` resolves citations that share a book -- "§§ 34, 35 BauGB" --
    and splitting first breaks them apart. Worth ~6% more norms (486 against 458).
    """
    out, seen = [], set()
    for c in refs_mod.canonicalise(refs_mod.extract('\n'.join(lines))):
        key = (refs_mod.corpus_key(c.book), c.section.lower())
        if key in idx and key not in seen:
            seen.add(key)
            out.append(idx[key])
    return out


def weighted_rrf(dense, lexical, w, k, rrf_k=60):
    """Fuse the two channels with an explicit weight instead of LanceDB's default.

    ``w`` is the dense channel's share: 1.0 is dense-only, 0.0 lexical-only.
    """
    score, rows = {}, {}
    for lst, weight in ((dense, w), (lexical, 1.0 - w)):
        for rank, r in enumerate(lst):
            key = (r.get('law_book'), r.get('paragraph'))
            score[key] = score.get(key, 0.0) + weight / (rrf_k + rank)
            rows.setdefault(key, r)
    return [rows[key] for key in sorted(score, key=score.get, reverse=True)[:k]]


def route_state(rows, share, k):
    """Reserve ``share`` of the budget for state-law norms, order preserved.

    Only a re-allocation: it can promote a Bavarian norm retrieval already found
    but cannot conjure one it missed. That ceiling is the point of measuring it.
    """
    # partitioned by position, not by value: these are dicts, and `r not in state`
    # would compare them field by field for every row
    is_state = [refs_mod.jurisdiction(str(r.get('law_book', ''))) == 'state' for r in rows]
    state = [r for r, s in zip(rows, is_state) if s]
    other = [r for r, s in zip(rows, is_state) if not s]
    want = int(round(k * share))
    out = state[:want] + other[:k - min(want, len(state))]
    return out[:k]


def ranked_norms(ret, queries, cfg, k=K_MAX, reranker=None, cite_idx=None,
                 defer_rerank=False, ls_ret=None):
    """The ranked, de-duplicated norm keys one case retrieves under ``cfg``."""
    subs = queries[cfg['query']]
    if subs is None:      # this case produced no query; it scores zero recall
        return [], []
    if cfg['merge'] == 'concat':
        subs = [' '.join(subs)]
    # Drop queries that sanitise away to nothing.  ``Retriever.search`` has no
    # guard for this (``LawRetriever._perform_search`` does), and an empty string
    # reaches the embedder as a zero-element tensor, which LanceDB then retries
    # seven times with exponential backoff -- a stalled sweep rather than an
    # error.  The rubric variant triggers it: stripping citations can empty a
    # line completely.
    subs = [s for s in subs if ret._clean(s).strip()]
    if not subs:
        return [], []
    # Re-ranking reorders a pool; it can only promote a norm that retrieval
    # already found, so the pool has to be deeper than the budget being scored
    # or there is nothing for it to promote into the top k.
    pool = cfg.get('pool') or (RERANK_POOL if cfg.get('rerank') else k)

    # Citations the rewriter named, resolved by exact lookup rather than search.
    cited = cited_rows(cite_idx, queries.get('rw_citations') or []) if cfg.get('cite_filter') else []
    # Norms a retrieved Leitsatz names, resolved the same way.  Same mechanism as
    # ``cited``, different source: the model's memory against a court's own chain.
    hopped = (leitsatz_rows(ls_ret, cite_idx, subs, cfg.get('ls_depth') or LS_DEPTH,
                            kinds=cfg.get('ls_kinds'))
              if cfg.get('two_hop') else [])
    # Whatever leads, leads once: a norm reached by both routes is not repeated.
    # Model-named citations go first where both are on -- they are the more
    # precise channel (~43% of them land in the top ten against ~18% for search),
    # and the Leitsatz chains are the broader one.
    lead, lead_seen = [], set()
    for r in cited + hopped:
        key = (r.get('law_book'), r.get('paragraph'))
        if key not in lead_seen:
            lead_seen.add(key)
            lead.append(r)
    if cfg.get('cite_filter') == 'only' or cfg.get('two_hop') == 'only':
        rows = lead[:k]
    else:
        if cfg.get('fusion_w') is not None:
            lists = [weighted_rrf(ret.search(s, pool, 'vector', cfg['drop_structural']),
                                  ret.search(s, pool, 'fts', cfg['drop_structural']),
                                  cfg['fusion_w'], pool) for s in subs]
        else:
            lists = [ret.search(s, pool, cfg['mode'], cfg['drop_structural']) for s in subs]
        rows = merge_lists(lists, pool, cfg['merge'])
        if cfg.get('rerank') and defer_rerank:
            # phase 1 of the two-phase path: hand the pool back unranked, so the
            # embedder can be unloaded before the re-ranker is loaded
            return [slim(r) for r in rows], ' '.join(subs)
        if cfg.get('rerank'):
            rows = apply_rerank(reranker, ' '.join(subs), rows, k)
        if cfg.get('state_share'):
            rows = route_state(rows, cfg['state_share'], k)
        if lead:
            # the exactly-resolved norms lead, search fills the rest of the budget
            rows = lead + [r for r in rows
                           if (r.get('law_book'), r.get('paragraph')) not in lead_seen]
            rows = rows[:k]
    out, seen = [], set()
    for r in rows:
        n = norm_of(r)
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return out, rows


# --------------------------------------------------------------------------
# scoring

def score(ranked_by_case, golds, structural_frac, cfg, name, seconds):
    """recall@k for every k and both gold sets, from one retrieval."""
    rows = []
    for gold_name, gold in golds.items():
        for k in K_SWEEP:
            hits = tot = st_hits = st_tot = 0
            per_case = []
            for i, g in gold.items():
                if not g:
                    continue
                got = set(ranked_by_case[i][:k])
                h = len(got & g)
                hits += h
                tot += len(g)
                per_case.append(h > 0)
                st = {n for n in g if refs_mod.jurisdiction(n[0]) == 'state'}
                if st:
                    st_hits += len(got & st)
                    st_tot += len(st)
            rows.append(dict(config=name, **cfg, gold_set=gold_name, k=k,
                             recall=hits / max(1, tot),
                             cases_with_hit=float(np.mean(per_case)) if per_case else 0.0,
                             state_recall=(st_hits / st_tot) if st_tot else np.nan,
                             hits=hits, gold=tot, structural=structural_frac,
                             seconds=round(seconds, 1)))
    return rows


def print_table(df, gold_set='gold', k=50, k_head=10):
    """The running comparison, newest numbers included.

    Two budgets side by side: ``k_head`` shows whether the top of the ranking is
    any good, ``k`` whether the pipeline can reach volume.  They can disagree --
    a configuration that wins on bulk recall may be worse at the head, which is
    what the generator actually reads first.
    """
    at = lambda kk: (df[(df.gold_set == gold_set) & (df.k == kk)]
                     .drop_duplicates('config', keep='last').set_index('config'))
    sel, head = at(k), at(k_head)
    if not len(sel):
        return
    b = float(sel.recall.get('baseline', np.nan))
    print(f'\n=== gold set "{gold_set}" '
          f'({"delta vs baseline" if not np.isnan(b) else "no baseline yet"}) ===', flush=True)
    hdr = (f'{"config":<26}{f"recall@{k_head}":>11}{f"recall@{k}":>11}{"delta":>9}'
           f'{f"any-hit@{k_head}":>13}{f"any-hit@{k}":>12}{f"state-law@{k}":>14}{"struct":>8}')
    print(hdr)
    print('-' * len(hdr))
    for cfg, r in sel.sort_values('recall', ascending=False).iterrows():
        d = '' if np.isnan(b) or cfg == 'baseline' else f'{100 * (r.recall - b):+.2f}'
        st = '--' if pd.isna(r.state_recall) else f'{100 * r.state_recall:.2f}%'
        pct = lambda v: '--' if v is None or np.isnan(v) else f'{100 * v:.2f}%'
        print(f'{cfg:<26}{pct(head.recall.get(cfg, np.nan)):>11}{100 * r.recall:10.2f}%{d:>9}'
              f'{pct(head.cases_with_hit.get(cfg, np.nan)):>13}'
              f'{100 * r.cases_with_hit:11.1f}%{st:>14}{100 * r.structural:7.1f}%')
    print('  recall@k = share of the case\'s cited norms retrieved within k passages'
          '\n  any-hit@k = share of cases with at least one cited norm retrieved'
          '\n  state-law@k = recall@k restricted to Bavarian state-law citations')


# --------------------------------------------------------------------------

def main(only=None, device='cuda', results=RESULTS, table_only=False, leitsatz=LEITSATZ):
    if table_only:
        print_table(pd.read_csv(results))
        return

    d = json.load(open(GPBAM))
    order = sorted(d['solutions'], key=int)
    facts = [d['facts'][k] for k in order]
    sols = [d['solutions'][k] for k in order]
    rubric = {i: d['rubric'][k] for i, k in enumerate(order) if k in d.get('rubric', {})}
    rewrites = json.load(open(REWRITES))

    golds = {
        'refex': {i: refex_norms(s) for i, s in enumerate(sols)},
        'gold': {i: {(refs_mod.corpus_key(c.book), c.section.lower())
                     for c in refs_mod.canonicalise(refs_mod.extract(s))}
                 for i, s in enumerate(sols)},
    }
    for n, g in golds.items():
        print(f'gold set "{n}": {np.mean([len(v) for v in g.values()]):.1f} norms per case')
    done = set()
    if os.path.exists(results):
        done = set(pd.read_csv(results).config.unique())
        print(f'{len(done)} configuration(s) already measured; skipping them')

    todo = [n for n in (only or VARIANTS) if n not in done]
    # only pay for the rewrites a pending configuration actually needs.  Keyed on
    # the ``query`` each configuration asks for, not on the variant's name: the
    # two used to coincide, because only ``rw_<name>`` variants read an
    # alternative prompt, but ``ls_q_*`` searches the Leitsatz index with one
    # under a different name and would silently have got no rewrites at all.
    wanted = {dict(BASELINE, **VARIANTS[t])['query'] for t in todo}
    need_alt = {n for n in REWRITE_PROMPTS if f'rw_{n}' in wanted}
    if any(dict(BASELINE, **VARIANTS[t]).get('cite_filter') for t in todo):
        need_alt.add('citations')     # the cite_* variants read the named norms
    alt = {n: make_rewrites(facts, n) for n in need_alt}
    queries = build_queries(facts, sols, rubric, rewrites, alt)
    print(f'{len(todo)} configuration(s) to run: {", ".join(todo) or "(none)"}\n', flush=True)

    # Only ever hold one retriever.  Caching one per corpus kept two embedding
    # models resident and OOM'd an 8 GB card that was also serving a generation
    # run; the configurations are ordered so this costs one reload, not many.
    todo.sort(key=lambda n: dict(BASELINE, **VARIANTS[n])['corpus'] != BASELINE['corpus'])
    ret, ret_kb, cite_idx, ls_ret = None, None, None, None
    for n, name in enumerate(todo, 1):
      # One configuration that cannot load or run should cost that
      # configuration, not the rest of the sweep -- a missing pad token in a
      # single re-ranker used to take every model queued behind it.
      try:
        cfg = dict(BASELINE, **VARIANTS[name])
        kb = COMBINED if cfg['corpus'] == 'combined' else FEDERAL
        # The dense model is only used for mode='vector'; hybrid and fts are
        # embedded by LanceDB's own registered function, so most configurations
        # never need it loaded at all.
        # fusion_w runs its own vector+fts pair through weighted_rrf, so it needs
        # the dense model even though its mode is 'hybrid'. Gating on mode alone
        # is why fusion_w25/50/75/90 have never produced a number.
        want_vectors = cfg['mode'] == 'vector' or cfg.get('fusion_w') is not None
        # Hand the dense model back as soon as a configuration stops needing it.
        # Only mode='vector' uses it -- hybrid and fts are embedded by LanceDB's
        # own registered function -- and leaving it resident cost ~2.4 GiB of an
        # 8 GB card that is also serving a generation run.
        if ret is not None and not want_vectors and ret.model is not None:
            ret.model = None
            free_gpu()
        if ret is None or kb != ret_kb or (want_vectors and ret.model is None):
            if ret is not None:
                ret = None
                free_gpu()
            if want_vectors:
                wait_for_gpu(1500)
            ret = Retriever(kb=kb, device=device, need_vectors=want_vectors)
            ret_kb = kb
            cite_idx = None
        # Skip only when the builder is inapplicable to the whole dataset.  An
        # individual case that yields nothing is a gap in that case, not grounds
        # to discard the variant -- keyed on "all", one empty case out of 81 used
        # to take the entire configuration with it.
        empty = [i for i in queries if queries[i][cfg['query']] is None]
        if len(empty) == len(queries):
            print(f'[{n}/{len(todo)}] {name}: SKIPPED -- the "{cfg["query"]}" builder '
                  'produces no queries for this dataset', flush=True)
            continue
        if empty:
            print(f'[{n}/{len(todo)}] {name}: {len(empty)} case(s) yield no query '
                  f'and score 0 recall: {empty}', flush=True)
        print(f'[{n}/{len(todo)}] {name}: {cfg}', flush=True)

        # Both the model's citations and a Leitsatz's Normenkette are resolved
        # against the same index, so either one needs it built.
        if (cfg.get('cite_filter') or cfg.get('two_hop')) and cite_idx is None:
            print('    building the citation index ...', flush=True)
            cite_idx = citation_index(ret)
            print(f'    {len(cite_idx)} norms addressable by citation', flush=True)
        if cfg.get('two_hop') and ls_ret is None:
            # need_vectors=False: hybrid and fts are embedded by the table's own
            # registered function, so the Leitsatz index costs no extra VRAM.
            ls_ret = Retriever(kb=leitsatz, device=device, need_vectors=False)
            print(f'    Leitsatz index: {ls_ret.table.count_rows()} rules', flush=True)

        t0 = time.time()
        ranked, struct, tot_rows = {}, 0, 0
        reranker = None

        if cfg.get('rerank'):
            # The deferred path applies the re-ranker and nothing after it, so a
            # configuration that also wanted state routing or citation-first
            # ordering would silently lose them. Fail instead.
            if cfg.get('state_share') or cfg.get('cite_filter') or cfg.get('two_hop'):
                raise NotImplementedError(
                    f'{name}: rerank combined with state_share/cite_filter/two_hop '
                    'is not wired through the two-phase path')
            # Two phases, so the two models never have to be resident together:
            # retrieve every pool with the embedder, park the embedder on the CPU,
            # then load the re-ranker. Peak becomes max(embedder, re-ranker) rather
            # than their sum, which is the difference between fitting on this card
            # beside a generation job and not.
            pools, qtexts = {}, {}
            for i in range(len(facts)):
                pools[i], qtexts[i] = ranked_norms(ret, queries[i], cfg,
                                                   cite_idx=cite_idx, defer_rerank=True)
                if (i + 1) % 40 == 0:
                    print(f'    retrieved {i + 1}/{len(facts)} pools', flush=True)
            n_pool = sum(len(p) for p in pools.values())
            remote = cfg['rerank'] in REMOTE_RERANKERS
            freed = [] if remote else unload_query_embedder(ret)
            print(f'    {n_pool} candidates retrieved; query embedder unloaded '
                  f'({", ".join(freed) or "nothing"}), {gpu_free_mib()} MiB free',
                  flush=True)

            if not remote:
                wait_for_gpu(2800)
            print(f"    loading re-ranker {RERANKERS[cfg['rerank']]}"
                  f"{' (remote)' if remote else ''} ...", flush=True)
            reranker = load_reranker(cfg['rerank'], device=device)
            for i in range(len(facts)):
                rows = apply_rerank(reranker, qtexts[i], pools[i], K_MAX) if pools[i] else []
                ranked[i] = [n for n in dict.fromkeys(
                    x for x in (norm_of(r) for r in rows) if x)]
                struct += sum(is_structural(r) for r in rows)
                tot_rows += len(rows)
                if (i + 1) % 20 == 0:
                    el = time.time() - t0
                    print(f'    re-ranked {i + 1}/{len(facts)}, {el / (i + 1):.1f}s/case',
                          flush=True)
        else:
            for i in range(len(facts)):
                ranked[i], rows = ranked_norms(ret, queries[i], cfg, cite_idx=cite_idx,
                                               ls_ret=ls_ret)
                struct += sum(is_structural(r) for r in rows)
                tot_rows += len(rows)
                if (i + 1) % 20 == 0:
                    el = time.time() - t0
                    print(f'    {i + 1}/{len(facts)} cases, {el / (i + 1):.1f}s/case, '
                          f'~{el / (i + 1) * (len(facts) - i - 1) / 60:.0f} min left',
                          flush=True)
        secs = time.time() - t0

        del reranker
        free_gpu()
        rows = score(ranked, golds, struct / max(1, tot_rows), cfg, name, secs)
        # Written per configuration, so an interrupted battery keeps everything it
        # has already paid for.  Read-concat-write rather than append: adding a
        # knob to BASELINE adds a column, and appending a wider row to a narrower
        # file produces a CSV that no longer parses at all.
        new = pd.DataFrame(rows)
        if os.path.exists(results):
            new = pd.concat([pd.read_csv(results), new], ignore_index=True)
        new.to_csv(results + '.tmp', index=False)
        os.replace(results + '.tmp', results)
        print_table(pd.read_csv(results))
      except Exception as e:
        print(f'!! {name} failed: {type(e).__name__}: {e}', flush=True)
        reranker = None
        free_gpu()


    print(f'\nwritten to {results}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--only', nargs='*', default=None, choices=list(VARIANTS))
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--table', action='store_true', help='re-print the table, no retrieval')
    ap.add_argument('--results', default=RESULTS,
                    help='where to read and write the measurements; point a rerun at '
                         'its own file rather than overwriting rows measured against '
                         'a different index')
    ap.add_argument('--leitsatz-index', default=LEITSATZ,
                    help='the Leitsatz index the two-hop configurations resolve '
                         'through; v2 carries 71,207 rules against v1 38,966')
    a = ap.parse_args()
    main(a.only, a.device, results=a.results, table_only=a.table,
         leitsatz=a.leitsatz_index)
