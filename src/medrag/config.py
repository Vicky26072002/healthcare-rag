"""Runtime settings, read from environment variables (optionally a .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")


def _path(name: str, default: str) -> Path:
    value = Path(os.getenv(name, default))
    return value if value.is_absolute() else PROJECT_ROOT / value


@dataclass
class Settings:
    data_dir: Path
    index_dir: Path
    external_dir: Path = PROJECT_ROOT / "data" / "external"
    sources: tuple[str, ...] = ("local",)
    embedding_backend: str = "sentence-transformers"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 5
    retrieval_mode: str = "hybrid"
    rerank: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_candidates: int = 30

    def __post_init__(self) -> None:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.retrieval_mode not in {"hybrid", "dense", "sparse"}:
            raise ValueError(f"Unknown retrieval mode: {self.retrieval_mode}")
        unknown = set(self.sources) - {"local", "medquad"}
        if unknown:
            raise ValueError(f"Unknown sources: {sorted(unknown)}")

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            data_dir=_path("MEDRAG_DATA_DIR", "data/raw"),
            index_dir=_path("MEDRAG_INDEX_DIR", "storage/index"),
            external_dir=_path("MEDRAG_EXTERNAL_DIR", "data/external"),
            sources=tuple(
                s.strip() for s in os.getenv("MEDRAG_SOURCES", "local,medquad").split(",") if s.strip()
            ),
            embedding_backend=os.getenv("MEDRAG_EMBEDDING_BACKEND", "sentence-transformers"),
            embedding_model=os.getenv(
                "MEDRAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            chunk_size=int(os.getenv("MEDRAG_CHUNK_SIZE", "800")),
            chunk_overlap=int(os.getenv("MEDRAG_CHUNK_OVERLAP", "120")),
            top_k=int(os.getenv("MEDRAG_TOP_K", "5")),
            retrieval_mode=os.getenv("MEDRAG_RETRIEVAL_MODE", "hybrid"),
            rerank=os.getenv("MEDRAG_RERANK", "true").lower() in {"1", "true", "yes"},
            reranker_model=os.getenv("MEDRAG_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            rerank_candidates=int(os.getenv("MEDRAG_RERANK_CANDIDATES", "30")),
        )
