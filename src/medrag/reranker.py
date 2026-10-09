"""Cross-encoder re-ranking: scores (question, passage) pairs jointly for higher precision."""

from __future__ import annotations


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder

        self.name = model_name
        self.model = CrossEncoder(model_name)

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        scores = self.model.predict([(query, p) for p in passages], batch_size=32, show_progress_bar=False)
        return [float(s) for s in scores]
