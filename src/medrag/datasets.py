"""Open medical datasets: MedQuAD (knowledge base) and TREC 2017 LiveQA-Med (benchmark).

MedQuAD - 47,457 QA pairs from 12 NIH websites (Ben Abacha & Demner-Fushman, 2019), CC BY 4.0.
          Answers of 3 MedlinePlus subsets were removed by the authors for copyright reasons.
LiveQA  - 104 real consumer health questions (TREC 2017 LiveQA medical task) with
          2,479 MedQuAD answers manually judged 1-Incorrect .. 4-Excellent.
"""

from __future__ import annotations

import csv
import io
import logging
import re
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from medrag.loaders import Document

logger = logging.getLogger(__name__)

MEDQUAD_ZIP = "https://github.com/abachaa/MedQuAD/archive/refs/heads/master.zip"
LIVEQA_ZIP = "https://github.com/abachaa/LiveQA_MedicalTask_TREC2017/archive/refs/heads/master.zip"
LIVEQA_ANSWERS_ZIP = "QA-TestSet-LiveQA-Med-Qrels-2479-Answers.zip"

# The qrels file abbreviates one collection name differently from the XML `source` attribute.
_SOURCE_ALIASES = {"MPlusHerbsSupplements": "MPlusHerbsSuppls"}
_GRADES = {"1-Incorrect": 0, "2-Related": 1, "3-Incomplete": 2, "4-Excellent": 3}
_CSV_ANSWER = re.compile(r"^Question:\s*(?P<q>.*?)\n(?:URL:\s*(?P<url>\S*)\s*\n)?Answer:\s*(?P<a>.*)$", re.S)


def answer_id(source: str, doc_id: str, pid: str) -> str:
    return f"{_SOURCE_ALIASES.get(source, source)}_{doc_id}_Sec{pid}"


def _clean(text: str) -> str:
    return " ".join(text.split())


# --------------------------------------------------------------------------- download
def _download_zip(url: str, target: Path) -> None:
    """Download a GitHub archive and extract it into `target` (top-level folder stripped)."""
    logger.info("Downloading %s", url)
    with urllib.request.urlopen(url, timeout=120) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    target.mkdir(parents=True, exist_ok=True)
    root = target.resolve()
    for member in archive.infolist():
        parts = Path(member.filename).parts[1:]
        if not parts or member.is_dir():
            continue
        dest = (target / Path(*parts)).resolve()
        if not dest.is_relative_to(root):  # zip-slip guard
            raise ValueError(f"Unsafe path in archive: {member.filename}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(archive.read(member))


def download_datasets(external_dir: Path) -> None:
    medquad = external_dir / "MedQuAD"
    if not (medquad / "readme.txt").exists():
        _download_zip(MEDQUAD_ZIP, medquad)
    liveqa = external_dir / "liveqa"
    if not (liveqa / "LiveQA_MedicalTask_TREC2017" / "TestDataset").exists():
        _download_zip(LIVEQA_ZIP, liveqa / "LiveQA_MedicalTask_TREC2017")
    if not any(liveqa.glob("*/All-qrels*.txt")):
        with zipfile.ZipFile(medquad / LIVEQA_ANSWERS_ZIP) as archive:
            for member in archive.namelist():
                if not member.startswith("__MACOSX") and not member.endswith("/"):
                    dest = liveqa / member
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(archive.read(member))
    logger.info("Datasets ready in %s", external_dir)


# --------------------------------------------------------------------------- MedQuAD
def _medquad_documents(path: Path) -> list[Document]:
    root = ET.parse(path).getroot()
    source = root.get("source", "")
    doc_id = root.get("id", path.stem)
    focus = _clean(root.findtext("Focus") or "")
    url = root.get("url", "")
    docs = []
    for pair in root.iter("QAPair"):
        question_el = pair.find("Question")
        answer = _clean(pair.findtext("Answer") or "")
        if question_el is None or not answer:
            continue
        docs.append(
            Document(
                doc_id=answer_id(source, doc_id, pair.get("pid", "")),
                text=answer,
                metadata={
                    "source": f"MedQuAD/{source}",
                    "section": _clean(question_el.text or ""),
                    "focus": focus,
                    "qtype": question_el.get("qtype", ""),
                    "url": url,
                },
            )
        )
    return docs


def load_medquad(medquad_dir: Path) -> list[Document]:
    """One Document per QA pair that still has an answer. Section = the MedQuAD question."""
    if not medquad_dir.exists():
        raise FileNotFoundError(f"MedQuAD not found at {medquad_dir}. Run `medrag download` first.")
    docs: list[Document] = []
    skipped = 0
    for path in sorted(medquad_dir.glob("*/*.xml")):
        try:
            docs.extend(_medquad_documents(path))
        except ET.ParseError:
            skipped += 1
    if skipped:
        logger.warning("Skipped %d malformed MedQuAD files", skipped)
    return docs


def load_judged_answers(liveqa_dir: Path) -> list[Document]:
    """The 2,479 judged answers ship with full text, including subsets removed from MedQuAD."""
    paths = list(liveqa_dir.glob("*/All-2479-Answers*.csv"))
    if not paths:
        return []
    docs = []
    with open(paths[0], encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            match = _CSV_ANSWER.match(row["Answer"].strip())
            if not match:
                continue
            answer = _clean(match.group("a")).removesuffix(")").strip()
            docs.append(
                Document(
                    doc_id=row["AnswerID"].removesuffix(".txt"),
                    text=answer,
                    metadata={
                        "source": "MedQuAD/" + row["AnswerID"].split("_", 1)[0],
                        "section": _clean(match.group("q")),
                        "url": match.group("url") or "",
                    },
                )
            )
    return docs


def load_medquad_corpus(external_dir: Path) -> list[Document]:
    """MedQuAD answers plus judged LiveQA answers, de-duplicated by answer ID."""
    docs = {d.doc_id: d for d in load_medquad(external_dir / "MedQuAD")}
    for doc in load_judged_answers(external_dir / "liveqa"):
        docs.setdefault(doc.doc_id, doc)
    return list(docs.values())


# --------------------------------------------------------------------------- LiveQA benchmark
@dataclass
class BenchmarkQuery:
    qid: str
    question: str
    paraphrase: str
    relevance: dict[str, int] = field(default_factory=dict)  # answer ID -> graded gain 0..3


def load_liveqa_benchmark(liveqa_dir: Path) -> list[BenchmarkQuery]:
    """LiveQA-Med 2017 test questions joined with the MedQuAD answer judgments."""
    xml_path = liveqa_dir / "LiveQA_MedicalTask_TREC2017" / "TestDataset" / (
        "TREC-2017-LiveQA-Medical-Test-Questions-w-summaries.xml"
    )
    qrels_paths = list(liveqa_dir.glob("*/All-qrels*.txt"))
    if not xml_path.exists() or not qrels_paths:
        raise FileNotFoundError(f"LiveQA benchmark not found in {liveqa_dir}. Run `medrag download`.")

    queries: dict[str, BenchmarkQuery] = {}
    for q in ET.parse(xml_path).getroot().iter("NLM-QUESTION"):
        original = q.find("Original-Question")
        subject = _clean(original.findtext("SUBJECT") or "") if original is not None else ""
        message = _clean(original.findtext("MESSAGE") or "") if original is not None else ""
        qid = q.get("qid", "").removeprefix("TQ")
        queries[qid] = BenchmarkQuery(
            qid=qid,
            question=f"{subject}. {message}" if subject else message,
            paraphrase=_clean(q.findtext("NIST-PARAPHRASE") or ""),
        )

    with open(qrels_paths[0], encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) == 3 and parts[0] in queries:
                queries[parts[0]].relevance[parts[2].removesuffix(".txt")] = _GRADES[parts[1]]
    return [q for q in queries.values() if q.relevance]
