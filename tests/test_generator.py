from medrag.chunking import Chunk
from medrag.generator import ExtractiveGenerator
from medrag.retriever import RetrievedChunk


def _result(text: str, **metadata) -> RetrievedChunk:
    return RetrievedChunk(chunk=Chunk("c", "d", text, metadata), score=1.0)


class KeywordScorer:
    """Stand-in for the cross-encoder: rewards texts containing 'dengue' and 'avoided'."""

    def __init__(self):
        self.seen: list[str] = []

    def score(self, query, texts):
        self.seen.extend(texts)
        return [5.0 * ("Dengue" in t) + 5.0 * ("avoided" in t) - 1.0 for t in texts]


def test_scorer_sees_title_and_section_context():
    scorer = KeywordScorer()
    results = [
        _result("Take ibuprofen every 6 to 8 hours for fever.", section="Fever"),
        _result("NSAIDs such as ibuprofen should be avoided.", title="Dengue Fever", section="Treatment"),
    ]
    answer = ExtractiveGenerator(max_sentences=1, scorer=scorer).generate("ibuprofen in dengue?", results)
    assert answer == "NSAIDs such as ibuprofen should be avoided. [2]"
    assert "Dengue Fever - Treatment: NSAIDs such as ibuprofen should be avoided." in scorer.seen


def test_fragments_and_duplicates_are_skipped():
    text = "diabetes, formerly called juvenile diabetes. Insulin is used to treat type 1 diabetes."
    results = [_result(text), _result("Insulin is used to treat type 1 diabetes.")]
    answer = ExtractiveGenerator(max_sentences=3).generate("How is type 1 diabetes treated with insulin?", results)
    assert answer == "Insulin is used to treat type 1 diabetes. [1]"


def test_no_confident_answer():
    results = [_result("Completely unrelated sentence about weather.")]
    assert "couldn't find" in ExtractiveGenerator().generate("dengue treatment", results)
