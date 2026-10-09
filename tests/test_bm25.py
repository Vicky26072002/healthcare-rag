import math

import pytest

from medrag.bm25 import SparseBM25

CORPUS = [
    "dengue fever spread by aedes mosquito".split(),
    "asthma inhaler relieves wheezing".split(),
    "high blood pressure and salt intake".split(),
    "mosquito nets prevent malaria".split(),
]


def _reference_score(query, doc, corpus, k1=1.5, b=0.75):
    avgdl = sum(len(d) for d in corpus) / len(corpus)
    score = 0.0
    for term in query:
        df = sum(term in d for d in corpus)
        idf = math.log1p((len(corpus) - df + 0.5) / (df + 0.5))
        tf = doc.count(term)
        score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(doc) / avgdl))
    return score


def test_scores_match_bm25_formula():
    query = "dengue mosquito".split()
    scores = SparseBM25(CORPUS).get_scores(query)
    for doc, score in zip(CORPUS, scores):
        assert score == pytest.approx(_reference_score(query, doc, CORPUS), rel=1e-5)


def test_top_n_ranks_and_filters_zero_scores():
    bm25 = SparseBM25(CORPUS)
    assert bm25.top_n("dengue mosquito".split(), 4) == [0, 3]
    assert bm25.top_n(["xyz"], 3) == []
