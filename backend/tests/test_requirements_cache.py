"""Cache des exigences extraites : même offre -> mêmes exigences, sans nouvel appel au LLM."""

import asyncio

import pytest

import app.requirements as requirements_module
from app.config import Settings
from app.requirements import (
    ExtractedRequirement,
    ExtractedRequirements,
    LLMProviderError,
    RequirementsCache,
    extract_requirements_cached,
    normalize_offer,
    requirements_cache_key,
)
from app.schemas import JobRequirements
from tests.conftest import FakeStructuredChatModel, make_settings
from tests.test_debug_match import EXTRACTED as MATCH_EXTRACTED
from tests.test_debug_match import _client

OFFER = "Développeur backend.\nTu maîtrises Python et Docker. Anglais apprécié."
OTHER_OFFER = "Data engineer. Tu maîtrises Spark et Airflow."


def _extracted(*labels: str) -> ExtractedRequirements:
    return ExtractedRequirements(
        job_title="Poste",
        requirements=[
            ExtractedRequirement(
                label=label,
                category="compétence technique",
                importance="obligatoire",
                search_queries=[label, f"{label} projet"],
            )
            for label in labels
        ],
    )


def _run(
    offer: str, model: FakeStructuredChatModel, settings: Settings, cache: RequirementsCache | None
) -> tuple[JobRequirements, bool]:
    return asyncio.run(extract_requirements_cached(offer, model, settings, None, cache))


def _labels(job: JobRequirements) -> list[str]:
    return [r.label for r in job.requirements]


class TestExtractRequirementsCached:
    def test_same_offer_is_served_from_cache_without_llm_call(self, settings: Settings) -> None:
        cache = RequirementsCache(8)
        model = FakeStructuredChatModel(responses=[_extracted("Python", "Docker"), _extracted("Go")])

        first, first_cached = _run(OFFER, model, settings, cache)
        second, second_cached = _run(OFFER, model, settings, cache)

        assert (first_cached, second_cached) == (False, True)
        assert second == first and _labels(second) == ["Python", "Docker"]
        assert len(model.calls) == 1

    def test_layout_differences_hit_the_same_entry(self, settings: Settings) -> None:
        cache = RequirementsCache(8)
        model = FakeStructuredChatModel(responses=[_extracted("Python")])
        _run(OFFER, model, settings, cache)

        reformatted = "  " + OFFER.replace("\n", "\r\n\r\n").replace(" ", "  ") + "\n"
        _, cached = _run(reformatted, model, settings, cache)

        assert cached and len(model.calls) == 1

    def test_different_offer_calls_the_llm(self, settings: Settings) -> None:
        cache = RequirementsCache(8)
        model = FakeStructuredChatModel(responses=[_extracted("Python"), _extracted("Spark")])

        _run(OFFER, model, settings, cache)
        other, cached = _run(OTHER_OFFER, model, settings, cache)

        assert not cached and _labels(other) == ["Spark"]
        assert len(model.calls) == 2 and len(cache) == 2

    def test_prompt_version_change_invalidates_entries(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cache = RequirementsCache(8)
        model = FakeStructuredChatModel(responses=[_extracted("Python"), _extracted("Python", "Docker")])
        _run(OFFER, model, settings, cache)

        monkeypatch.setattr(
            requirements_module, "EXTRACTION_PROMPT_VERSION", requirements_module.EXTRACTION_PROMPT_VERSION + 1
        )
        job, cached = _run(OFFER, model, settings, cache)

        assert not cached and _labels(job) == ["Python", "Docker"]
        assert len(model.calls) == 2

    def test_extraction_model_change_invalidates_entries(self, settings: Settings) -> None:
        cache = RequirementsCache(8)
        model = FakeStructuredChatModel(responses=[_extracted("Python"), _extracted("Python")])
        _run(OFFER, model, settings, cache)

        _, cached = _run(OFFER, model, make_settings(llm_model="autre-modele"), cache)

        assert not cached and len(model.calls) == 2

    def test_failures_are_not_cached(self, settings: Settings) -> None:
        cache = RequirementsCache(8)
        model = FakeStructuredChatModel(responses=[ConnectionError("x"), ConnectionError("x"), _extracted("Python")])

        with pytest.raises(LLMProviderError):
            _run(OFFER, model, settings, cache)
        job, cached = _run(OFFER, model, settings, cache)

        assert not cached and _labels(job) == ["Python"] and len(cache) == 1

    def test_without_cache_always_calls_the_llm(self, settings: Settings) -> None:
        model = FakeStructuredChatModel(responses=[_extracted("Python"), _extracted("Python")])

        results = [_run(OFFER, model, settings, None) for _ in range(2)]

        assert [cached for _, cached in results] == [False, False] and len(model.calls) == 2

    def test_size_zero_disables_the_cache(self, settings: Settings) -> None:
        cache = RequirementsCache(0)
        model = FakeStructuredChatModel(responses=[_extracted("Python"), _extracted("Python")])

        _run(OFFER, model, settings, cache)
        _, cached = _run(OFFER, model, settings, cache)

        assert not cached and len(cache) == 0


class TestRequirementsCache:
    def _job(self, title: str) -> JobRequirements:
        return JobRequirements(job_title=title, requirements=[])

    def test_least_recently_used_entry_is_evicted(self) -> None:
        cache = RequirementsCache(2)
        cache.put("a", self._job("A"))
        cache.put("b", self._job("B"))
        assert cache.get("a") is not None  # "a" redevient la plus récente

        cache.put("c", self._job("C"))

        assert cache.get("b") is None
        assert cache.get("a") is not None and cache.get("c") is not None and len(cache) == 2

    def test_entries_are_isolated_copies(self) -> None:
        cache = RequirementsCache(2)
        job = self._job("A")
        cache.put("a", job)
        job.job_title = "modifié"

        served = cache.get("a")
        assert served is not None and served.job_title == "A"
        served.job_title = "modifié aussi"
        assert cache.get("a").job_title == "A"  # type: ignore[union-attr]


class TestCacheKey:
    def test_is_a_sha256_hex_digest(self, settings: Settings) -> None:
        key = requirements_cache_key(OFFER, settings)

        assert len(key) == 64 and int(key, 16) >= 0
        assert "Python" not in key  # aucune donnée de l'offre en clair

    def test_depends_on_offer_model_version_and_limit(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        base = requirements_cache_key(OFFER, settings)

        assert requirements_cache_key(OFFER, settings) == base
        assert requirements_cache_key(OTHER_OFFER, settings) != base
        assert requirements_cache_key(OFFER, make_settings(llm_model="autre")) != base
        assert requirements_cache_key(OFFER, make_settings(max_requirements=5)) != base
        monkeypatch.setattr(requirements_module, "EXTRACTION_PROMPT_VERSION", 999)
        assert requirements_cache_key(OFFER, settings) != base

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("  a\r\n\r\n b\t c  ", "a b c"),
            ("Expérience", "Expérience"),  # NFD -> NFC
        ],
    )
    def test_offer_normalization(self, raw: str, expected: str) -> None:
        assert normalize_offer(raw) == expected


def test_debug_match_reports_whether_requirements_came_from_cache(settings: Settings, sectioned_cv_pdf: bytes) -> None:
    extraction = FakeStructuredChatModel(responses=[MATCH_EXTRACTED])
    with _client(settings, extraction) as client:
        reports = [
            client.post(
                "/debug/match",
                files={"file": ("cv.pdf", sectioned_cv_pdf, "application/pdf")},
                data={"job_offer": OFFER},
            ).json()["report"]
            for _ in range(2)
        ]

    assert [r["requirements_from_cache"] for r in reports] == [False, True]
    assert len(extraction.calls) == 1
    assert reports[0]["requirements"] == reports[1]["requirements"]
