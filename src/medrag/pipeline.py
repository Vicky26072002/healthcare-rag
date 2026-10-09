"""End-to-end RAG pipeline: ingest -> index -> retrieve -> generate."""

from __future__ import annotations

import logging

from medrag.chunking import chunk_documents
from medrag.config import Settings
from medrag.datasets import load_medquad_corpus
from medrag.embeddings import Embedder, get_embedder
from medrag.generator import Answer, ExtractiveGenerator, to_sources
from medrag.loaders import Document, load_documents
from medrag.retriever import HybridRetriever
from medrag.safety import DISCLAIMER, detect_emergency
from medrag.vector_store import VectorStore

logger = logging.getLogger(__name__)


def load_sources(settings: Settings) -> list[Document]:
    docs: list[Document] = []
    if "local" in settings.sources:
        docs.extend(load_documents(settings.data_dir))
    if "medquad" in settings.sources:
        docs.extend(load_medquad_corpus(settings.external_dir))
    return docs


def make_retriever(store: VectorStore, embedder: Embedder, settings: Settings) -> HybridRetriever:
    reranker = None
    if settings.rerank:
        from medrag.reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker(settings.reranker_model)
    return HybridRetriever(
        store,
        embedder,
        settings.retrieval_mode,
        reranker=reranker,
        rerank_candidates=settings.rerank_candidates,
    )


class RAGPipeline:
    def __init__(self, retriever: HybridRetriever, settings: Settings, generator=None):
        self.retriever = retriever
        self.settings = settings
        self.generator = generator or ExtractiveGenerator(scorer=retriever.reranker)

    @classmethod
    def build(cls, settings: Settings, embedder: Embedder | None = None) -> "RAGPipeline":
        """Load documents, chunk, embed and persist a fresh index."""
        embedder = embedder or get_embedder(settings.embedding_backend, settings.embedding_model)
        docs = load_sources(settings)
        chunks = chunk_documents(docs, settings.chunk_size, settings.chunk_overlap)
        logger.info("Loaded %d documents -> %d chunks", len(docs), len(chunks))
        if not chunks:
            raise ValueError(f"No content found in {settings.data_dir}")

        store = VectorStore(dim=embedder.dim, embedding_name=embedder.name)
        logger.info("Embedding %d chunks with %s ...", len(chunks), embedder.name)
        store.add(chunks, embedder.encode([c.embed_text for c in chunks]))
        store.save(settings.index_dir)
        logger.info("Index saved to %s", settings.index_dir)
        return cls(make_retriever(store, embedder, settings), settings)

    @classmethod
    def load(cls, settings: Settings, embedder: Embedder | None = None) -> "RAGPipeline":
        store = VectorStore.load(settings.index_dir)
        embedder = embedder or get_embedder(settings.embedding_backend, settings.embedding_model)
        if store.embedding_name and store.embedding_name != embedder.name:
            raise ValueError(
                f"Index was built with '{store.embedding_name}' but current embedder is "
                f"'{embedder.name}'. Re-run `medrag ingest`."
            )
        return cls(make_retriever(store, embedder, settings), settings)

    def ask(
        self, question: str, top_k: int | None = None, mode: str | None = None, rerank: bool | None = None
    ) -> Answer:
        results = self.retriever.retrieve(
            question, k=top_k or self.settings.top_k, mode=mode, rerank=rerank
        )
        return Answer(
            question=question,
            answer=self.generator.generate(question, results),
            sources=to_sources(results),
            emergency_flags=detect_emergency(question),
            disclaimer=DISCLAIMER,
        )
