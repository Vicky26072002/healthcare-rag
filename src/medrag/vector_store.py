"""Persistent vector store. Uses FAISS when installed, otherwise exact NumPy search."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from medrag.chunking import Chunk

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover - depends on environment
    faiss = None


class VectorStore:
    def __init__(self, dim: int, embedding_name: str = ""):
        self.dim = dim
        self.embedding_name = embedding_name
        self.chunks: list[Chunk] = []
        self.vectors = np.zeros((0, dim), dtype="float32")
        self._index = faiss.IndexFlatIP(dim) if faiss is not None else None

    def __len__(self) -> int:
        return len(self.chunks)

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        vectors = np.ascontiguousarray(vectors, dtype="float32")
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"expected vectors of shape (n, {self.dim})")
        self.chunks.extend(chunks)
        self.vectors = np.vstack([self.vectors, vectors])
        if self._index is not None:
            self._index.add(vectors)

    def search(self, query_vector: np.ndarray, k: int = 5) -> list[tuple[int, float]]:
        """Return (chunk position, cosine similarity) pairs, best first."""
        if not self.chunks:
            return []
        k = min(k, len(self.chunks))
        query = np.ascontiguousarray(query_vector, dtype="float32").reshape(1, -1)
        if self._index is not None:
            scores, ids = self._index.search(query, k)
            return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]
        scores = self.vectors @ query[0]
        top = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in top]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self.vectors)
        with open(directory / "chunks.jsonl", "w", encoding="utf-8") as f:
            for chunk in self.chunks:
                f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
        meta = {"dim": self.dim, "embedding": self.embedding_name, "count": len(self.chunks)}
        (directory / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> "VectorStore":
        meta_path = directory / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"No index found in {directory}. Run `medrag ingest` first.")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        store = cls(dim=meta["dim"], embedding_name=meta.get("embedding", ""))
        with open(directory / "chunks.jsonl", encoding="utf-8") as f:
            chunks = [Chunk.from_dict(json.loads(line)) for line in f if line.strip()]
        store.add(chunks, np.load(directory / "vectors.npy"))
        return store
