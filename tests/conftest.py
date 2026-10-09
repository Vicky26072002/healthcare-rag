from pathlib import Path

import pytest

from medrag.config import Settings
from medrag.embeddings import HashingEmbedder
from medrag.pipeline import RAGPipeline

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=DATA_DIR,
        index_dir=tmp_path / "index",
        embedding_backend="hashing",
        chunk_size=500,
        chunk_overlap=80,
    )


@pytest.fixture
def pipeline(settings: Settings) -> RAGPipeline:
    # Hashing embedder keeps tests fast and offline (no model download).
    return RAGPipeline.build(settings, embedder=HashingEmbedder())
