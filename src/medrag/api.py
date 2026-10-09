"""FastAPI service exposing the RAG pipeline."""

from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from medrag import __version__
from medrag.config import Settings
from medrag.pipeline import RAGPipeline

app = FastAPI(title="MedRAG - Healthcare RAG API", version=__version__)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    top_k: int = Field(5, ge=1, le=20)
    mode: str = Field("hybrid", pattern="^(hybrid|dense|sparse)$")
    rerank: bool = True


@lru_cache(maxsize=1)
def get_pipeline() -> RAGPipeline:
    return RAGPipeline.load(Settings.from_env())


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.post("/ask")
def ask(request: AskRequest) -> dict:
    try:
        pipeline = get_pipeline()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    rerank = request.rerank and pipeline.retriever.reranker is not None
    answer = pipeline.ask(request.question, top_k=request.top_k, mode=request.mode, rerank=rerank)
    return asdict(answer)
