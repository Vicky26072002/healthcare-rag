from medrag.embeddings import HashingEmbedder
from medrag.evaluation import evaluate, load_qa
from medrag.pipeline import RAGPipeline
from medrag.safety import detect_emergency
from medrag.vector_store import VectorStore

from conftest import DATA_DIR


def test_index_is_persisted_and_reloadable(pipeline, settings):
    store = VectorStore.load(settings.index_dir)
    assert len(store) == len(pipeline.retriever.store) > 0
    reloaded = RAGPipeline.load(settings, embedder=HashingEmbedder())
    assert reloaded.ask("dengue mosquito").sources


def test_retrieval_finds_relevant_source(pipeline):
    for mode in ("hybrid", "dense", "sparse"):
        results = pipeline.retriever.retrieve("warning signs of severe dengue", k=3, mode=mode)
        assert results[0].chunk.metadata["source"] == "dengue.md", mode


def test_ask_returns_cited_answer(pipeline):
    answer = pipeline.ask("Should I avoid ibuprofen in dengue?")
    assert "[1]" in answer.answer or "[2]" in answer.answer
    assert answer.disclaimer
    assert answer.sources[0].source == "dengue.md"


def test_emergency_detection():
    assert detect_emergency("I have chest pain and shortness of breath") == [
        "chest pain",
        "shortness of breath",
    ]
    assert detect_emergency("what is a normal HbA1c?") == []


def test_hybrid_retrieval_quality(pipeline):
    qa = load_qa(DATA_DIR.parent / "eval" / "qa.jsonl")
    result = evaluate(pipeline.retriever, qa, k=5, mode="hybrid")
    assert result.hit_rate >= 0.8
