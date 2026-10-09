"""Section-aware text chunking with character-level overlap."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from medrag.loaders import Document

_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*$")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    metadata: dict = field(default_factory=dict)

    @property
    def embed_text(self) -> str:
        """Text used for embedding/indexing: document title and section give useful context."""
        parts = (self.metadata.get("title"), self.metadata.get("section"))
        context = " - ".join(dict.fromkeys(p for p in parts if p))
        return f"{context}. {self.text}" if context else self.text

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "text": self.text,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        return cls(**data)


def split_sections(text: str) -> list[tuple[str | None, str]]:
    """Split markdown text into (heading, body) pairs. Plain text yields one section."""
    sections: list[tuple[str | None, str]] = []
    heading: str | None = None
    lines: list[str] = []
    for line in text.splitlines():
        match = _HEADING.match(line)
        if match:
            if "\n".join(lines).strip():
                sections.append((heading, "\n".join(lines)))
            heading, lines = match.group(1), []
        else:
            lines.append(line)
    if "\n".join(lines).strip():
        sections.append((heading, "\n".join(lines)))
    return sections


def _units(text: str, size: int) -> list[str]:
    """Break text into paragraphs, falling back to sentences for long paragraphs."""
    units: list[str] = []
    for para in _PARAGRAPH_BREAK.split(text):
        para = " ".join(para.split())
        if not para:
            continue
        if len(para) <= size:
            units.append(para)
        else:
            units.extend(s for s in _SENTENCE_END.split(para) if s)
    return units


def _overlap_tail(text: str, overlap: int) -> str:
    if overlap <= 0 or len(text) <= overlap:
        return text if overlap > 0 else ""
    tail = text[-overlap:]
    # Start the overlap on a word boundary.
    return tail.split(" ", 1)[1] if " " in tail else tail


def chunk_text(text: str, size: int = 800, overlap: int = 120) -> list[str]:
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    chunks: list[str] = []
    current = ""
    for unit in _units(text, size):
        if len(unit) > size:
            step = size - overlap
            pieces = [unit[i : i + size] for i in range(0, len(unit), step)]
        else:
            pieces = [unit]
        for piece in pieces:
            candidate = f"{current} {piece}" if current else piece
            if len(candidate) <= size:
                current = candidate
                continue
            chunks.append(current)
            tail = _overlap_tail(current, overlap)
            current = f"{tail} {piece}" if tail else piece
            if len(current) > size:
                current = piece
    if current:
        chunks.append(current)
    return chunks


def chunk_documents(docs: list[Document], size: int = 800, overlap: int = 120) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in docs:
        index = 0
        for section, body in split_sections(doc.text):
            for piece in chunk_text(body, size, overlap):
                metadata = dict(doc.metadata)
                metadata["chunk_index"] = index
                if section:
                    metadata["section"] = section
                chunks.append(
                    Chunk(chunk_id=f"{doc.doc_id}::{index}", doc_id=doc.doc_id, text=piece, metadata=metadata)
                )
                index += 1
    return chunks
