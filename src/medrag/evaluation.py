"""Retrieval evaluation.

- `evaluate`: Hit Rate@k / MRR on the small local QA set (relevance = expected source file).
- `evaluate_benchmark`: TREC 2017 LiveQA-Med consumer questions with graded expert judgments
  over MedQuAD answers -> Hit@1, Hit@5, MRR@10, nDCG@10 at the answer (document) level.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

from medrag.datasets import BenchmarkQuery
from medrag.retriever import HybridRetriever


@dataclass
class EvalResult:
    mode: str
    k: int
    hit_rate: float
    mrr: float
    n: int


def load_qa(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def evaluate(retriever: HybridRetriever, qa: list[dict], k: int = 5, mode: str = "hybrid") -> EvalResult:
    hits, reciprocal_ranks = 0, 0.0
    for item in qa:
        results = retriever.retrieve(item["question"], k=k, mode=mode, rerank=False)
        sources = [r.chunk.metadata.get("source") for r in results]
        if item["source"] in sources:
            hits += 1
            reciprocal_ranks += 1.0 / (sources.index(item["source"]) + 1)
    n = len(qa) or 1
    return EvalResult(mode=mode, k=k, hit_rate=hits / n, mrr=reciprocal_ranks / n, n=len(qa))


@dataclass
class BenchmarkResult:
    name: str
    hit_at_1: float
    hit_at_5: float
    mrr_at_10: float
    ndcg_at_10: float
    n: int
    ms_per_query: float


def _dcg(gains: list[int]) -> float:
    return sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(gains))


def rank_documents(retriever: HybridRetriever, query: str, k: int, mode: str, rerank: bool) -> list[str]:
    """Retrieve chunks and collapse them to a ranked list of unique document IDs."""
    doc_ids: list[str] = []
    for r in retriever.retrieve(query, k=k * 3, mode=mode, rerank=rerank):
        if r.chunk.doc_id not in doc_ids:
            doc_ids.append(r.chunk.doc_id)
    return doc_ids[:k]


def evaluate_benchmark(
    retriever: HybridRetriever,
    queries: list[BenchmarkQuery],
    mode: str = "hybrid",
    rerank: bool = False,
    min_grade: int = 2,
    use_paraphrase: bool = False,
    k: int = 10,
) -> BenchmarkResult:
    """A retrieved answer counts as relevant when its judged grade >= `min_grade`
    (2 = "Incomplete", 3 = "Excellent"). nDCG uses the graded judgments directly.
    Unjudged answers are treated as non-relevant (standard TREC practice)."""
    scored = [q for q in queries if any(g >= min_grade for g in q.relevance.values())]
    hit1 = hit5 = rr = ndcg = 0.0
    start = time.perf_counter()
    for q in scored:
        text = q.paraphrase if use_paraphrase and q.paraphrase else q.question
        ranked = rank_documents(retriever, text, k, mode, rerank)
        relevant = [i for i, d in enumerate(ranked) if q.relevance.get(d, 0) >= min_grade]
        if relevant:
            first = relevant[0]
            hit1 += first == 0
            hit5 += first < 5
            rr += 1.0 / (first + 1)
        ideal = _dcg(sorted(q.relevance.values(), reverse=True)[:k])
        ndcg += _dcg([q.relevance.get(d, 0) for d in ranked]) / ideal if ideal else 0.0
    elapsed = time.perf_counter() - start
    n = len(scored) or 1
    name = f"{mode}{' + rerank' if rerank else ''}"
    return BenchmarkResult(
        name=name,
        hit_at_1=hit1 / n,
        hit_at_5=hit5 / n,
        mrr_at_10=rr / n,
        ndcg_at_10=ndcg / n,
        n=len(scored),
        ms_per_query=1000 * elapsed / n,
    )
