"""Plancher des certifications de langue (règle du code, pas du prompt) : une preuve valide
qui cite un score ou un niveau officiel dans la langue demandée vaut au moins « démontré »."""

import asyncio

import pytest

from app.certifications import certified_languages, proves_language, requirement_language
from app.llm_calls import RateLimiter
from app.matcher import evaluate_requirement, verify_assessment
from app.schemas import Chunk, Requirement, SearchResult
from tests.conftest import FakeStructuredChatModel, make_settings
from tests.test_matcher import TODAY, _assess


def _language(label: str, queries: list[str] | None = None) -> Requirement:
    return Requirement(
        id="R1",
        label=label,
        category="langue",
        importance="obligatoire",
        search_queries=queries or [label, f"{label} courant"],
    )


ENGLISH = _language("Anglais professionnel", ["anglais professionnel", "english"])
GERMAN = _language("Allemand courant", ["allemand", "german"])


class TestDetection:
    @pytest.mark.parametrize(
        ("quote", "languages"),
        [
            ("Anglais : B1 (TOEIC 790)", ["anglais"]),
            ("TOEFL iBT 95", ["anglais"]),
            ("IELTS 7,5", ["anglais"]),
            ("Cambridge C1 Advanced", ["anglais"]),
            ("Français : DELF B2", ["francais"]),
            ("DALF C1", ["francais"]),
            ("Allemand : B1, Goethe-Zertifikat", ["allemand"]),
            ("Espagnol : DELE B2", ["espagnol"]),
            ("Italien : B2 (score 82/100)", ["italien"]),
            ("Portugais – C1 (92 %)", ["portugais"]),
            # Un segment par langue : la certification de l'anglais ne vaut pas pour l'allemand.
            ("Français : langue maternelle · Anglais : B1 (TOEIC 790) · Allemand : notions", ["anglais"]),
        ],
    )
    def test_official_scores_and_levels(self, quote: str, languages: list[str]) -> None:
        assert certified_languages(quote) == languages

    @pytest.mark.parametrize(
        "quote",
        [
            "Anglais : B2",  # niveau seulement déclaré
            "Anglais courant, séjour de six mois à Dublin",
            "Anglais B2 (2023)",  # une année n'est pas un score
            "Préparation au TOEIC en cours",  # pas de score
            "Master à l'université de Cambridge",  # pas un examen
            "Sept. 2023 – Août 2025 · Lyon",
        ],
    )
    def test_not_official(self, quote: str) -> None:
        assert certified_languages(quote) == []

    def test_requirement_language(self) -> None:
        assert requirement_language(ENGLISH) == "anglais"
        assert requirement_language(_language("Langue étrangère", ["langue étrangère", "bilingue"])) is None
        assert requirement_language(_language("Anglais et allemand", ["anglais", "allemand"])) is None

    def test_certification_must_be_in_the_requested_language(self) -> None:
        quote = "Anglais : B1 (TOEIC 790) · Allemand : notions"

        assert proves_language(quote, ENGLISH)
        assert not proves_language(quote, GERMAN)
        # Exigence sans langue nommée : toute certification officielle compte.
        assert proves_language(quote, _language("Langue étrangère", ["langue étrangère", "bilingue"]))


CHUNK = "LANGUES\nFrançais : langue maternelle · Anglais : B1 (TOEIC 790) · Allemand : notions"


class TestFloor:
    def test_mentioned_language_with_certification_becomes_demonstrated(self) -> None:
        assessment = _assess("mentionné", (0, "Anglais : B1 (TOEIC 790)"))

        result = verify_assessment(assessment, ENGLISH, {0: CHUNK}, TODAY)

        assert (result.level, result.llm_level, result.certification_floor) == ("démontré", "mentionné", True)

    def test_higher_levels_are_kept_without_flag(self) -> None:
        result = verify_assessment(_assess("solide", (0, "Anglais : B1 (TOEIC 790)")), ENGLISH, {0: CHUNK}, TODAY)

        assert (result.level, result.certification_floor) == ("solide", False)

    def test_certification_in_another_language_does_not_raise(self) -> None:
        result = verify_assessment(_assess("mentionné", (0, "Allemand : notions")), GERMAN, {0: CHUNK}, TODAY)

        assert (result.level, result.certification_floor) == ("mentionné", False)

    def test_self_declared_level_is_not_raised(self) -> None:
        chunk = "LANGUES\nAnglais : B2 · Espagnol : A2"

        result = verify_assessment(_assess("mentionné", (0, "Anglais : B2")), ENGLISH, {0: chunk}, TODAY)

        assert (result.level, result.certification_floor) == ("mentionné", False)

    def test_only_language_requirements_are_raised(self) -> None:
        skill = Requirement(
            id="R1", label="Anglais technique", category="compétence technique", importance="souhaitée",
            search_queries=["anglais", "documentation en anglais"],
        )  # fmt: skip

        result = verify_assessment(_assess("mentionné", (0, "Anglais : B1 (TOEIC 790)")), skill, {0: CHUNK}, TODAY)

        assert (result.level, result.certification_floor) == ("mentionné", False)

    def test_an_invalid_quote_raises_nothing(self) -> None:
        # Citation introuvable dans le chunk : rejetée, donc aucune preuve valide à relever.
        result = verify_assessment(_assess("mentionné", (0, "Anglais : C1 (TOEIC 980)")), ENGLISH, {0: CHUNK}, TODAY)

        assert (result.level, result.downgraded, result.certification_floor) == ("non démontré", True, False)

    def test_points_follow_the_raised_level(self) -> None:
        settings = make_settings()
        chunk = Chunk(section="LANGUES", index=0, text=CHUNK)
        llm = FakeStructuredChatModel(responses=[_assess("mentionné", (0, "Anglais : B1 (TOEIC 790)"))])

        result = asyncio.run(
            evaluate_requirement(
                ENGLISH,
                [SearchResult(chunk=chunk, score=0.9)],
                llm,
                RateLimiter.from_settings(settings),
                settings,
                TODAY,
            )
        )

        assert (result.level, result.points, result.certification_floor) == ("démontré", 75, True)
