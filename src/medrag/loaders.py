"""Load source documents (.md, .txt, .pdf) from disk."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf"}
_TITLE = re.compile(r"^#\s+(.+?)\s*$", re.M)


@dataclass
class Document:
    doc_id: str
    text: str
    metadata: dict = field(default_factory=dict)


def _load_pdf(path: Path) -> list[Document]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    docs = []
    for page_no, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            docs.append(
                Document(
                    doc_id=f"{path.stem}#p{page_no}",
                    text=text,
                    metadata={"source": path.name, "page": page_no},
                )
            )
    return docs


def load_file(path: Path) -> list[Document]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(path)
    if suffix in {".md", ".txt"}:
        text = path.read_text(encoding="utf-8")
        metadata = {"source": path.name}
        title = _TITLE.search(text)
        if title:
            metadata["title"] = title.group(1)
        return [Document(doc_id=path.stem, text=text, metadata=metadata)]
    raise ValueError(f"Unsupported file type: {path}")


def load_documents(data_dir: Path) -> list[Document]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    docs: list[Document] = []
    for path in sorted(data_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            docs.extend(load_file(path))
    return docs
