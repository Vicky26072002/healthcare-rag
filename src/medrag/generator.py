"""Answer generation.

Phase 1 (current): an extractive generator that selects the most relevant sentences
from the retrieved context and cites them. No LLM required.

Phase 2 (planned): an LLM generator that consumes `build_prompt(...)`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from medrag.retriever import RetrievedChunk, tokenize

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_SENTENCE_START = re.compile(r"^[A-Z(\"'-]")

SYSTEM_PROMPT = (
    "You are a careful healthcare information assistant. Answer ONLY from the provided "
    "context. Cite sources as [n]. If the context does not contain the answer, say you "
    "don't know. Never give a diagnosis or prescribe medication doses; recommend seeing "
    "a qualified clinician."
)


@dataclass
class Source:
    index: int
    source: str
    section: str | None
    text: str
    score: float
    url: str = ""


@dataclass
class Answer:
    question: str
    answer: str
    sources: list[Source] = field(default_factory=list)
    emergency_flags: list[str] = field(default_factory=list)
    disclaimer: str = ""


def format_context(results: list[RetrievedChunk]) -> str:
    blocks = []
    for i, r in enumerate(results, start=1):
        meta = r.chunk.metadata
        header = meta.get("source", r.chunk.doc_id)
        if meta.get("section"):
            header += f" - {meta['section']}"
        blocks.append(f"[{i}] ({header})\n{r.chunk.text}")
    return "\n\n".join(blocks)


def build_prompt(question: str, results: list[RetrievedChunk]) -> str:
    """Prompt for the Phase 2 LLM generator."""
    return f"Context:\n{format_context(results)}\n\nQuestion: {question}\nAnswer (with citations):"


def to_sources(results: list[RetrievedChunk]) -> list[Source]:
    return [
        Source(
            index=i,
            source=r.chunk.metadata.get("source", r.chunk.doc_id),
            section=r.chunk.metadata.get("section"),
            text=r.chunk.text,
            score=round(r.rerank_score if r.rerank_score is not None else r.score, 4),
            url=r.chunk.metadata.get("url", ""),
        )
        for i, r in enumerate(results, start=1)
    ]


class ExtractiveGenerator:
    """Select the most relevant sentences from the retrieved context and cite their chunk.

    With a cross-encoder `scorer`, every sentence is scored jointly with the question;
    otherwise query-term overlap is used (fast, offline fallback).
    """

    def __init__(self, max_sentences: int = 3, scorer=None):
        self.max_sentences = max_sentences
        self.scorer = scorer
        # Below this the best sentence is not considered an answer.
        self.min_score = -3.0 if scorer is not None else 0.1

    @staticmethod
    def _sentences(results: list[RetrievedChunk]) -> list[tuple[int, int, str]]:
        """(context rank, position, sentence); drops fragments and duplicates."""
        seen: set[str] = set()
        out = []
        for ctx_idx, r in enumerate(results, start=1):
            for sent_idx, sentence in enumerate(_SENTENCE_END.split(r.chunk.text)):
                sentence = sentence.strip()
                key = sentence.lower()
                # Chunk overlap can start mid-sentence; skip such fragments.
                if len(sentence) < 20 or not _SENTENCE_START.match(sentence) or key in seen:
                    continue
                seen.add(key)
                out.append((ctx_idx, sent_idx, sentence))
        return out

    @staticmethod
    def _context(chunk) -> str:
        """Document title / section, e.g. "Dengue Fever - Treatment"."""
        meta = chunk.metadata
        parts = [meta.get("title"), meta.get("section")]
        return " - ".join(dict.fromkeys(p for p in parts if p))

    def _score(self, question: str, sentences: list[tuple[int, int, str]], results) -> list[float]:
        if self.scorer is not None:
            # Score each sentence with its context: "Aspirin and NSAIDs should be avoided" only
            # answers a dengue question when the scorer can see it comes from "Dengue - Treatment".
            texts = []
            for ctx, _, sentence in sentences:
                context = self._context(results[ctx - 1].chunk)
                texts.append(f"{context}: {sentence}" if context else sentence)
            scores = self.scorer.score(question, texts)
            # Small prior towards higher-ranked contexts.
            return [score - 0.5 * (ctx - 1) for score, (ctx, _, _) in zip(scores, sentences)]
        query_terms = set(tokenize(question))
        scores = []
        for ctx, _, sentence in sentences:
            overlap = len(query_terms & set(tokenize(sentence))) / (len(query_terms) or 1)
            scores.append(overlap + 0.05 / ctx)
        return scores

    def generate(self, question: str, results: list[RetrievedChunk]) -> str:
        sentences = self._sentences(results)
        if not sentences:
            return "I couldn't find relevant information in the knowledge base."
        scores = self._score(question, sentences, results)
        scored = sorted(zip(scores, sentences), key=lambda x: x[0], reverse=True)
        best = scored[: self.max_sentences]
        if best[0][0] < self.min_score:
            return "I couldn't find a confident answer in the knowledge base."
        # Keep reading order: by context rank, then position in the chunk.
        best.sort(key=lambda x: (x[1][0], x[1][1]))
        return " ".join(f"{sentence} [{ctx}]" for _, (ctx, _, sentence) in best)


class LLMGenerator:
    """Placeholder for Phase 2 - plug an LLM in here using SYSTEM_PROMPT + build_prompt."""

    def generate(self, question: str, results: list[RetrievedChunk]) -> str:
        raise NotImplementedError("LLM generation is planned for Phase 2.")
