import pytest

from medrag.chunking import chunk_documents, chunk_text, split_sections
from medrag.loaders import Document


def test_split_sections_tracks_headings():
    text = "# Title\nintro\n## Symptoms\nfever\n## Treatment\nrest"
    sections = split_sections(text)
    assert [h for h, _ in sections] == ["Title", "Symptoms", "Treatment"]


def test_chunks_respect_size_limit():
    text = " ".join(f"Sentence number {i} about health." for i in range(200))
    chunks = chunk_text(text, size=200, overlap=40)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)


def test_chunks_overlap():
    text = " ".join(f"Sentence {i}." for i in range(100))
    chunks = chunk_text(text, size=120, overlap=30)
    # The start of each chunk should repeat text from the end of the previous one.
    for prev, nxt in zip(chunks, chunks[1:]):
        assert nxt.split(" ")[0] in prev


def test_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_text("abc", size=10, overlap=10)


def test_chunk_documents_metadata():
    doc = Document(doc_id="d", text="## Diagnosis\nBlood tests.", metadata={"source": "d.md"})
    [chunk] = chunk_documents([doc])
    assert chunk.chunk_id == "d::0"
    assert chunk.metadata == {"source": "d.md", "chunk_index": 0, "section": "Diagnosis"}
    assert chunk.embed_text.startswith("Diagnosis.")
