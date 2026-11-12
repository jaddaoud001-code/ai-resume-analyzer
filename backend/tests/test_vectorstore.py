import pytest

from app.chunking import chunk_cv
from app.schemas import Chunk
from app.vectorstore import CVIndex
from tests.conftest import GENERIC_CV_TEXT, KeywordEmbeddings


@pytest.fixture
def index() -> CVIndex:
    return CVIndex(chunk_cv(GENERIC_CV_TEXT), KeywordEmbeddings())


def test_most_relevant_chunk_comes_first(index: CVIndex) -> None:
    results = index.search("React TypeScript", k=3)

    assert "Initech" in results[0].chunk.text
    assert results[0].chunk.section == "EXPÉRIENCE"


def test_scores_are_cosine_similarities_sorted_descending(index: CVIndex) -> None:
    scores = [r.score for r in index.search("déploiement cloud AWS", k=5)]

    assert scores == sorted(scores, reverse=True)
    assert all(-1.0 <= s <= 1.0 + 1e-6 for s in scores)


def test_identical_text_scores_one() -> None:
    chunk = Chunk(section="LANGUES", index=0, text="Français natif")
    index = CVIndex([chunk], KeywordEmbeddings())

    [result] = index.search("LANGUES\nFrançais natif", k=1)

    assert result.score == pytest.approx(1.0)
    assert result.chunk == chunk


def test_k_larger_than_index_returns_every_chunk(index: CVIndex) -> None:
    chunks = chunk_cv(GENERIC_CV_TEXT)

    results = index.search("Python", k=100)

    assert sorted(r.chunk.index for r in results) == [c.index for c in chunks]


def test_empty_chunk_list_is_rejected() -> None:
    with pytest.raises(ValueError):
        CVIndex([], KeywordEmbeddings())
