"""Vectorised Okapi BM25 using a precomputed sparse (documents x vocabulary) weight matrix.

Scoring a query is a sparse column sum, which is orders of magnitude faster than
looping over documents in Python (as `rank_bm25` does) on a corpus of ~40K chunks.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse


class SparseBM25:
    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.vocab: dict[str, int] = {}
        rows, cols, counts = [], [], []
        for row, tokens in enumerate(corpus):
            tf: dict[int, int] = {}
            for token in tokens:
                col = self.vocab.setdefault(token, len(self.vocab))
                tf[col] = tf.get(col, 0) + 1
            rows.extend([row] * len(tf))
            cols.extend(tf.keys())
            counts.extend(tf.values())

        n_docs = len(corpus)
        shape = (n_docs, max(len(self.vocab), 1))
        tf_matrix = sparse.csr_matrix((np.array(counts, dtype="float32"), (rows, cols)), shape=shape)

        doc_len = np.array([len(tokens) for tokens in corpus], dtype="float32")
        avg_len = float(doc_len.mean()) if n_docs else 0.0
        doc_freq = np.bincount(np.array(cols, dtype=np.int64), minlength=shape[1]).astype("float32")
        # Lucene-style IDF: always positive, unlike the classic Robertson-Sparck Jones form.
        self.idf = np.log1p((n_docs - doc_freq + 0.5) / (doc_freq + 0.5)).astype("float32")

        # weight(d, t) = idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * |d| / avgdl))
        norm = k1 * (1 - b + b * doc_len / avg_len) if avg_len else np.full(n_docs, k1, "float32")
        tf_coo = tf_matrix.tocoo()
        tf = tf_coo.data
        weights = self.idf[tf_coo.col] * tf * (k1 + 1) / (tf + norm[tf_coo.row])
        self.matrix = sparse.csc_matrix((weights, (tf_coo.row, tf_coo.col)), shape=shape)
        self.n_docs = n_docs

    def get_scores(self, query_tokens: list[str]) -> np.ndarray:
        cols = [self.vocab[t] for t in query_tokens if t in self.vocab]
        if not cols:
            return np.zeros(self.n_docs, dtype="float32")
        return np.asarray(self.matrix[:, cols].sum(axis=1)).ravel()

    def top_n(self, query_tokens: list[str], n: int) -> list[int]:
        scores = self.get_scores(query_tokens)
        n = min(n, self.n_docs)
        if n <= 0:
            return []
        top = np.argpartition(-scores, n - 1)[:n]
        top = top[np.argsort(-scores[top], kind="stable")]
        return [int(i) for i in top if scores[i] > 0]
