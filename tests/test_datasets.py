from pathlib import Path

from medrag.chunking import chunk_documents
from medrag.datasets import BenchmarkQuery, answer_id, load_liveqa_benchmark, load_medquad_corpus
from medrag.embeddings import HashingEmbedder
from medrag.evaluation import evaluate_benchmark
from medrag.retriever import HybridRetriever
from medrag.vector_store import VectorStore

MEDQUAD_XML = """<?xml version="1.0" encoding="UTF-8"?>
<Document id="0000001" source="NIDDK" url="https://example.org/acromegaly">
<Focus>Acromegaly</Focus>
<QAPairs>
  <QAPair pid="1">
    <Question qid="0000001-1" qtype="information">What is (are) Acromegaly ?</Question>
    <Answer>Acromegaly is a hormonal disorder caused by too much growth hormone.</Answer>
  </QAPair>
  <QAPair pid="2">
    <Question qid="0000001-2" qtype="treatment">What are the treatments for Acromegaly ?</Question>
    <Answer></Answer>
  </QAPair>
</QAPairs>
</Document>
"""

ANSWERS_CSV = (
    "AnswerID,Answer\n"
    'ADAM_0000009_Sec1.txt,"Question: What causes gout ?\n'
    "URL: https://example.org/gout\n"
    'Answer: Gout is caused by uric acid crystals in the joint.) \n"\n'
)

LIVEQA_XML = """<?xml version="1.0" encoding="UTF-8"?>
<LiveQA2017-Medical-Test-Set-Full>
  <NLM-QUESTION qid="TQ1">
    <Original-Question qfile="1.txt"><SUBJECT>gout</SUBJECT><MESSAGE>why does my toe hurt so much</MESSAGE></Original-Question>
    <NIST-PARAPHRASE>What causes gout?</NIST-PARAPHRASE>
  </NLM-QUESTION>
</LiveQA2017-Medical-Test-Set-Full>
"""


def _make_external(tmp_path: Path) -> Path:
    ext = tmp_path / "external"
    (ext / "MedQuAD" / "5_NIDDK_QA").mkdir(parents=True)
    (ext / "MedQuAD" / "5_NIDDK_QA" / "0000001.xml").write_text(MEDQUAD_XML, encoding="utf-8")
    judged = ext / "liveqa" / "QA-TestSet"
    judged.mkdir(parents=True)
    (judged / "All-2479-Answers-retrieved-from-MedQuAD.csv").write_text(ANSWERS_CSV, encoding="utf-8")
    (judged / "All-qrels_LiveQAMed2017.txt").write_text(
        "1 4-Excellent ADAM_0000009_Sec1.txt\n1 1-Incorrect NIDDK_0000001_Sec1.txt\n", encoding="utf-8"
    )
    test_dir = ext / "liveqa" / "LiveQA_MedicalTask_TREC2017" / "TestDataset"
    test_dir.mkdir(parents=True)
    (test_dir / "TREC-2017-LiveQA-Medical-Test-Questions-w-summaries.xml").write_text(LIVEQA_XML, encoding="utf-8")
    return ext


def test_answer_id_aliases():
    assert answer_id("MPlusHerbsSupplements", "0000001", "2") == "MPlusHerbsSuppls_0000001_Sec2"


def test_load_corpus_skips_empty_answers_and_merges_judged(tmp_path):
    docs = {d.doc_id: d for d in load_medquad_corpus(_make_external(tmp_path))}
    assert set(docs) == {"NIDDK_0000001_Sec1", "ADAM_0000009_Sec1"}
    assert docs["NIDDK_0000001_Sec1"].metadata["section"] == "What is (are) Acromegaly ?"
    assert docs["ADAM_0000009_Sec1"].text == "Gout is caused by uric acid crystals in the joint."


def test_liveqa_benchmark_and_metrics(tmp_path):
    ext = _make_external(tmp_path)
    [query] = load_liveqa_benchmark(ext / "liveqa")
    assert query.question == "gout. why does my toe hurt so much"
    assert query.relevance == {"ADAM_0000009_Sec1": 3, "NIDDK_0000001_Sec1": 0}

    embedder = HashingEmbedder()
    chunks = chunk_documents(load_medquad_corpus(ext))
    store = VectorStore(embedder.dim, embedder.name)
    store.add(chunks, embedder.encode([c.embed_text for c in chunks]))
    result = evaluate_benchmark(HybridRetriever(store, embedder), [query], use_paraphrase=True)
    assert result.n == 1
    assert result.hit_at_1 == 1.0 and result.ndcg_at_10 == 1.0


def test_queries_without_relevant_answers_are_excluded():
    q = BenchmarkQuery(qid="1", question="x", paraphrase="", relevance={"a": 0})
    store = VectorStore(4)
    result = evaluate_benchmark(HybridRetriever(store, HashingEmbedder(4)), [q])
    assert result.n == 0
