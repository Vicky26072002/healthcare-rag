"""Hybrid retrieval: dense (embeddings) + sparse (BM25), fused with Reciprocal Rank Fusion."""

from __future__ import annotations

import re
from dataclasses import dataclass

from medrag.bm25 import SparseBM25
from medrag.chunking import Chunk
from medrag.embeddings import Embedder
from medrag.vector_store import VectorStore

_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    """a an and are as at be by can do does for from how i in is it its may of on or
    should that the their this to was what when where which who why will with you your""".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None
    rerank_score: float | None = None


class HybridRetriever:
    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        mode: str = "hybrid",
        rrf_k: int = 60,
        candidate_multiplier: int = 4,
        reranker=None,
        rerank_candidates: int = 30,
    ):
        self.store = store
        self.embedder = embedder
        self.mode = mode
        self.rrf_k = rrf_k
        self.candidate_multiplier = candidate_multiplier
        self.reranker = reranker
        self.rerank_candidates = rerank_candidates
        corpus = [tokenize(c.embed_text) for c in store.chunks]
        self.bm25 = SparseBM25(corpus) if corpus else None

    def _dense(self, query: str, n: int) -> list[int]:
        vector = self.embedder.encode([query])[0]
        return [i for i, _ in self.store.search(vector, n)]

    def _sparse(self, query: str, n: int) -> list[int]:
        if self.bm25 is None:
            return []
        return self.bm25.top_n(tokenize(query), n)

    def retrieve(
        self, query: str, k: int = 5, mode: str | None = None, rerank: bool | None = None
    ) -> list[RetrievedChunk]:
        mode = mode or self.mode
        rerank = self.reranker is not None if rerank is None else rerank
        if rerank and self.reranker is None:
            raise ValueError("Re-ranking requested but no reranker is configured")
        if not query.strip() or len(self.store) == 0:
            return []
        pool = max(k, self.rerank_candidates) if rerank else k
        n = pool * self.candidate_multiplier
        dense = self._dense(query, n) if mode in {"hybrid", "dense"} else []
        sparse = self._sparse(query, n) if mode in {"hybrid", "sparse"} else []

        fused: dict[int, RetrievedChunk] = {}
        for ranking, attr in ((dense, "dense_rank"), (sparse, "sparse_rank")):
            for rank, pos in enumerate(ranking, start=1):
                item = fused.setdefault(pos, RetrievedChunk(chunk=self.store.chunks[pos], score=0.0))
                item.score += 1.0 / (self.rrf_k + rank)
                setattr(item, attr, rank)

        ranked = sorted(fused.values(), key=lambda r: r.score, reverse=True)[:pool]
        if rerank:
            scores = self.reranker.score(query, [r.chunk.embed_text for r in ranked])
            for item, score in zip(ranked, scores):
                item.rerank_score = score
            ranked.sort(key=lambda r: r.rerank_score, reverse=True)
        return ranked[:k]
